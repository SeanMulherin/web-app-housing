"""Read-only Supabase ledger check; does not admit visitors or reserve requests."""

import json

from housing_safety import SafetyBlocked, configured_store
from supabase_safety import SupabaseSafetyStore


def main():
    try:
        store = configured_store()
        if not isinstance(store, SupabaseSafetyStore):
            print('Selected backend is SQLite. Verify the original persistent disk and ledger file; do not reinitialize it.')
            return 2
        status = store.status()
    except SafetyBlocked as error:
        print(str(error))
        print('Inspect the safety RPC category in server logs. No RentCast request was made.')
        return 1
    print(json.dumps({
        'backend': 'supabase', 'status': 'ok',
        'attempt_count31d': status['attempt_count31d'],
        'attempt_count32d': status['attempt_count32d'],
    }))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
