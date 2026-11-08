from housing_safety import SafetyBlocked


TOKEN = 'health-monitor-token-with-at-least-32-characters'


class StatusStore:
    def __init__(self, error=None):
        self.error = error
        self.status_calls = 0

    def status(self):
        self.status_calls += 1
        if self.error is not None:
            raise self.error
        return {'allowed': True}


def test_health_check_fails_closed_when_server_token_is_missing(monkeypatch):
    import main

    monkeypatch.delenv('HOUSING_HEALTHCHECK_TOKEN', raising=False)
    monkeypatch.setattr(main, 'configured_store', lambda: (_ for _ in ()).throw(AssertionError('store must not be opened')))

    response = main.app.test_client().get('/health/housing-safety')

    assert response.status_code == 503
    assert response.data == b''
    assert response.headers['Cache-Control'] == 'no-store'


def test_health_check_rejects_missing_or_wrong_bearer_token_before_database_access(monkeypatch):
    import main

    monkeypatch.setenv('HOUSING_HEALTHCHECK_TOKEN', TOKEN)
    monkeypatch.setattr(main, 'configured_store', lambda: (_ for _ in ()).throw(AssertionError('store must not be opened')))
    client = main.app.test_client()

    assert client.get('/health/housing-safety').status_code == 401
    assert client.get('/health/housing-safety', headers={'Authorization': 'Bearer wrong-token'}).status_code == 401


def test_health_check_reads_status_once_and_returns_no_ledger_data(monkeypatch):
    import main

    store = StatusStore()
    monkeypatch.setenv('HOUSING_HEALTHCHECK_TOKEN', TOKEN)
    monkeypatch.setattr(main, 'configured_store', lambda: store)

    response = main.app.test_client().get(
        '/health/housing-safety',
        headers={'Authorization': f'Bearer {TOKEN}'},
    )

    assert response.status_code == 204
    assert response.data == b''
    assert response.headers['Cache-Control'] == 'no-store'
    assert store.status_calls == 1


def test_health_check_converts_safety_storage_failure_to_empty_503(monkeypatch):
    import main

    store = StatusStore(SafetyBlocked('do not expose this database detail'))
    monkeypatch.setenv('HOUSING_HEALTHCHECK_TOKEN', TOKEN)
    monkeypatch.setattr(main, 'configured_store', lambda: store)

    response = main.app.test_client().get(
        '/health/housing-safety',
        headers={'Authorization': f'Bearer {TOKEN}'},
    )

    assert response.status_code == 503
    assert response.data == b''
    assert b'do not expose' not in response.data
    assert store.status_calls == 1


def test_health_check_converts_store_configuration_failure_to_empty_503(monkeypatch):
    import main

    monkeypatch.setenv('HOUSING_HEALTHCHECK_TOKEN', TOKEN)
    monkeypatch.setattr(
        main,
        'configured_store',
        lambda: (_ for _ in ()).throw(SafetyBlocked('do not expose this configuration detail')),
    )

    response = main.app.test_client().get(
        '/health/housing-safety',
        headers={'Authorization': f'Bearer {TOKEN}'},
    )

    assert response.status_code == 503
    assert response.data == b''
    assert b'do not expose' not in response.data


def test_health_check_does_not_accept_post(monkeypatch):
    import main

    store = StatusStore()
    monkeypatch.setenv('HOUSING_HEALTHCHECK_TOKEN', TOKEN)
    monkeypatch.setattr(main, 'configured_store', lambda: store)

    response = main.app.test_client().post(
        '/health/housing-safety',
        headers={'Authorization': f'Bearer {TOKEN}'},
    )

    assert response.status_code == 405
    assert store.status_calls == 0
