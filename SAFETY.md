# RentCast billing and abuse safeguards

Every outbound RentCast attempt reserves one unit in a durable SQLite ledger
before networking. The cap covers all valuation, subject-listing, and neighborhood
pagination calls, including the JSON `/api/analysis` and HTML `/forecast` routes.
Concurrent workers share the counter; timeouts, failed requests, and interrupted
requests retain their reservations. Automatic HTTP redirects are disabled.

There is a fixed **$5 ceiling on estimated request fees from this app**, alongside
the configurable rolling 31-day request allowance. The spending ceiling cannot
be raised through environment settings. It treats every outbound attempt as a
billable **$0.20 request**, without assuming any unused free quota. This means at
most **25 attempts in any rolling 32 days**; the extra day covers a 31-day billing
month and daylight-saving changes. Failed attempts also consume this allowance.
Existing ledger reservations count toward both limits, without a reset or schema
migration. A lower configured request allowance still takes precedence.

The $0.20 upper bound comes from RentCast's published pricing, verified October 7,
2026: <https://www.rentcast.io/api>. Verify it against the actual subscription
before enabling requests and after any pricing change. The guard does not observe
RentCast invoices or payment activity. A rate above $0.20 requires a stricter
reservation cost before further requests can safely run.

The default allowance is **zero**. Missing, corrupt, locked, or uninitialized
storage blocks new analyses with HTTP 503; exhaustion returns HTTP 429. Runtime
code never creates an empty replacement ledger or falls back to memory.

## Production setup

1. Mount a persistent disk on the backend (for example `/var/data` on Render).
   All Gunicorn workers must use the same ledger. Use a single service instance
   with that shared disk; separate replicas or services need a shared transactional
   database instead of separate SQLite files. An ephemeral filesystem cannot
   preserve a budget through replacement deployments.
2. Set `HOUSING_SAFETY_DB_PATH=/var/data/housing-safety.sqlite3`.
3. Explicitly initialize that new ledger once, from the backend shell:

   ```sh
   python housing_safety.py init /var/data/housing-safety.sqlite3
   ```

   Initialization refuses an existing file. Do not delete or recreate a ledger to
   recover from an error or to replenish quota. Restore the original persistent
   storage and its ledger instead. Keep a backup with the disk's normal backups.
   An older backup omits reservations made after its snapshot: keep the allowance
   at zero until RentCast usage is reconciled before enabling a restored ledger.
4. Check RentCast's current billing-period usage and set
   `HOUSING_RENTCAST_MAX_REQUESTS_31D` to a conservative allowance no greater than
   your **remaining** included requests, less headroom for other authorized use.
   Leave it at `0` until that allowance is known. One address analysis can need up
   to six units. Raising this value authorizes additional attempts; it does not
   erase the previous 31 days of usage.
   Even if this setting exceeds 25, the independent $5 ceiling still blocks
   additional attempts in the rolling 32-day spending window.
5. Deploy the backend and verify the persistent file and settings survive a
   restart. An OPTIONS request is safe for checking reachability and CORS without
   triggering RentCast. Tests below use temporary ledgers and mocked requests.

This is an application request-fee ceiling, **not an account-wide payment cap**.
RentCast does not offer a provider-enforced dollar cap. Fixed subscription fees,
taxes, fees already incurred, other services, and direct use of the API key are
outside this ledger. A $5 total RentCast API bill requires the $0/month Developer
plan, no untracked subscription usage, and enough remaining budget for any
existing charges or taxes. Paid plans' fixed fees already exceed $5. Use a
dedicated server-side key and reconcile account usage before enabling requests.
The conservative guard usually permits fewer calls than the Developer plan's
50 free requests; it prioritizes the ceiling over consuming the full free quota.
Hosting and storage charges are also separate from RentCast request fees.

## Public admission limits

Both POST routes share these persistent limits, including cache hits:

| Setting | Default | Meaning |
| --- | --- | --- |
| `HOUSING_VISITOR_REQUESTS_PER_MINUTE` | `3` | Per visitor, rolling 60 seconds |
| `HOUSING_VISITOR_REQUESTS_PER_DAY` | `20` | Per visitor, rolling 24 hours |
| `HOUSING_GLOBAL_REQUESTS_PER_MINUTE` | `60` | Across all visitors, rolling 60 seconds |

Rate limits must be positive integers and cannot be disabled with zero. Invalid
configuration fails closed. Rate denials include `Retry-After`; blocked requests
do not extend the lockout. Visitor identifiers are salted hashes, and records
older than a day are removed as new requests are admitted.

The application uses the connection's `remote_addr` and ignores caller-supplied
forwarded headers. Behind a reverse proxy this may group visitors under the
proxy's address. Verify the deployed server's address behavior before adjusting
limits; do not blindly trust `X-Forwarded-For`. The global paid-request budget
applies regardless of visitor identification.

Public `force_refresh=true` requests return HTTP 403. Ordinary cached analyses
remain available within the visitor limits even when the paid allowance is zero.
The legacy forecast route remains subject to the paid cap even though it has no
analysis cache. Request bodies are limited to 8 KiB.

## Verification

```sh
python -m pytest -q
```

Tests cover concurrent processes, restart persistence, missing/uninitialized/
corrupt storage, reservation before networking, failed attempts, every RentCast
method, both public endpoints, public refresh denial, forwarded-header spoofing,
cache hits, and explicit 429/503 responses. No live RentCast requests are needed.
