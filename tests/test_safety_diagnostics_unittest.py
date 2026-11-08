import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from housing_safety import SafetyBlocked
from supabase_safety import SupabaseSafetyStore

URL = 'https://abcdefghijklmnopqrst.supabase.co'
KEY = 'sb_secret_do_not_log'
LEDGER = '12345678-1234-4234-8234-123456789abc'


class DiagnosticsTest(unittest.TestCase):
    def test_failures_remain_closed_and_logs_exclude_secrets(self):
        cases = [
            (HTTPError(URL, 401, KEY, {}, io.BytesIO(KEY.encode())), 'category=http status=401'),
            (URLError(KEY), 'category=transport'),
        ]
        for failure, category in cases:
            with self.subTest(category=category):
                store = SupabaseSafetyStore(URL, KEY, LEDGER)
                with patch.object(store._opener, 'open', side_effect=failure) as request:
                    with self.assertLogs('supabase_safety', level='WARNING') as logs:
                        with self.assertRaises(SafetyBlocked) as error:
                            store.reserve_rentcast_request()
                self.assertIn(category, logs.output[0])
                self.assertNotIn(KEY, str(logs.output) + str(error.exception))
                self.assertNotIn(URL, str(logs.output))
                self.assertEqual(request.call_count, 1)
                self.assertEqual(error.exception.status_code, 503)

    def test_read_only_check_uses_status_rpc(self):
        import diagnose_safety
        store = SupabaseSafetyStore(URL, KEY, LEDGER)
        result = {'attempt_count31d': 2, 'attempt_count32d': 3}
        with patch.object(diagnose_safety, 'configured_store', return_value=store):
            with patch.object(store, '_rpc', return_value=result) as rpc:
                with patch('sys.stdout', new_callable=io.StringIO) as output:
                    self.assertEqual(diagnose_safety.main(), 0)
        rpc.assert_called_once_with('housing_safety_status', {})
        self.assertEqual(json.loads(output.getvalue())['attempt_count32d'], 3)

    def test_invalid_response_is_logged_without_body(self):
        store = SupabaseSafetyStore(URL, KEY, LEDGER)
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.return_value = KEY.encode()
        with patch.object(store._opener, 'open', return_value=response) as request:
            with self.assertLogs('supabase_safety', level='WARNING') as logs:
                with self.assertRaises(SafetyBlocked):
                    store.check_visitor('offline-test')
        self.assertIn('category=invalid_response', logs.output[0])
        self.assertNotIn(KEY, str(logs.output))
        self.assertEqual(request.call_count, 1)


if __name__ == '__main__':
    unittest.main()
