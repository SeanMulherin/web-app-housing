-- Operator bootstrap only. Run once as the database owner (normally postgres).
-- An existing schema or RPC signature aborts this transaction without replacing
-- its ledger. Save the returned ledger_id in the backend configuration.
BEGIN;

CREATE SCHEMA housing_safety;
REVOKE ALL ON SCHEMA housing_safety FROM PUBLIC, anon, authenticated, service_role;

CREATE TABLE housing_safety.metadata (
    id smallint PRIMARY KEY CHECK (id = 1),
    version integer NOT NULL CHECK (version = 1),
    ledger_id uuid NOT NULL UNIQUE
);
CREATE TABLE housing_safety.rentcast_attempts (
    created_at timestamp with time zone NOT NULL
);
CREATE INDEX rentcast_attempts_time ON housing_safety.rentcast_attempts (created_at);
CREATE TABLE housing_safety.visitor_requests (
    visitor text NOT NULL CHECK (visitor ~ '^[0-9a-f]{64}$'),
    created_at timestamp with time zone NOT NULL
);
CREATE INDEX visitor_requests_time ON housing_safety.visitor_requests (created_at);
CREATE INDEX visitor_requests_visitor_time ON housing_safety.visitor_requests (visitor, created_at);

ALTER TABLE housing_safety.metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE housing_safety.rentcast_attempts ENABLE ROW LEVEL SECURITY;
ALTER TABLE housing_safety.visitor_requests ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON ALL TABLES IN SCHEMA housing_safety FROM PUBLIC, anon, authenticated, service_role;

INSERT INTO housing_safety.metadata (id, version, ledger_id)
VALUES (1, 1, pg_catalog.gen_random_uuid());

CREATE FUNCTION public.housing_safety_reserve(p_ledger_id uuid, p_max_requests integer DEFAULT 0)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $function$
DECLARE
    v_metadata housing_safety.metadata%ROWTYPE;
    v_now timestamp with time zone;
    v_count bigint;
    v_release timestamp with time zone;
    v_reason text := 'ok';
    v_retry integer := NULL;
BEGIN
    -- A stale repeatable-read snapshot could survive waiting for the row lock.
    -- PostgREST's normal read-committed mode takes fresh snapshots after locking.
    IF pg_catalog.current_setting('transaction_isolation') <> 'read committed' THEN
        RETURN pg_catalog.jsonb_build_object('version', 1, 'ledger_id', p_ledger_id,
            'allowed', false, 'reason', 'storage', 'retry_after', NULL);
    END IF;
    SELECT * INTO v_metadata FROM housing_safety.metadata WHERE id = 1 FOR UPDATE;
    IF NOT FOUND OR v_metadata.version <> 1 OR p_ledger_id IS NULL
       OR v_metadata.ledger_id <> p_ledger_id THEN
        v_reason := 'storage';
    ELSIF p_max_requests IS NULL OR p_max_requests < 0 THEN
        v_reason := 'settings';
    ELSE
        -- Acquire the singleton lock before reading time or counting attempts.
        v_now := pg_catalog.clock_timestamp();
        SELECT pg_catalog.count(*) INTO v_count FROM housing_safety.rentcast_attempts
        WHERE created_at > v_now - interval '768 hours';
        IF v_count >= 25 THEN
            -- The 25th newest attempt determines when another slot becomes available.
            SELECT created_at INTO v_release FROM housing_safety.rentcast_attempts
            WHERE created_at > v_now - interval '768 hours'
            ORDER BY created_at DESC OFFSET 24 LIMIT 1;
            v_retry := GREATEST(1, pg_catalog.ceil(
                EXTRACT(epoch FROM (v_release + interval '768 hours' - v_now)))::integer);
            v_reason := 'spend';
        ELSE
            SELECT pg_catalog.count(*) INTO v_count FROM housing_safety.rentcast_attempts
            WHERE created_at > v_now - interval '744 hours';
            IF v_count >= p_max_requests THEN
                v_reason := 'budget';
                IF p_max_requests > 0 THEN
                    SELECT created_at INTO v_release FROM housing_safety.rentcast_attempts
                    WHERE created_at > v_now - interval '744 hours'
                    ORDER BY created_at DESC OFFSET (p_max_requests - 1) LIMIT 1;
                    v_retry := GREATEST(1, pg_catalog.ceil(
                        EXTRACT(epoch FROM (v_release + interval '744 hours' - v_now)))::integer);
                END IF;
            ELSE
                INSERT INTO housing_safety.rentcast_attempts (created_at) VALUES (v_now);
            END IF;
        END IF;
    END IF;
    RETURN pg_catalog.jsonb_build_object('version', 1, 'ledger_id', p_ledger_id,
        'allowed', v_reason = 'ok', 'reason', v_reason, 'retry_after', v_retry);
EXCEPTION WHEN OTHERS THEN
    RETURN pg_catalog.jsonb_build_object('version', 1, 'ledger_id', p_ledger_id,
        'allowed', false, 'reason', 'storage', 'retry_after', NULL);
END;
$function$;

CREATE FUNCTION public.housing_safety_admit(
    p_ledger_id uuid, p_visitor text,
    p_visitor_per_minute integer DEFAULT 3,
    p_visitor_per_day integer DEFAULT 20,
    p_global_per_minute integer DEFAULT 60
)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $function$
DECLARE
    v_metadata housing_safety.metadata%ROWTYPE;
    v_now timestamp with time zone;
    v_count bigint;
    v_release timestamp with time zone;
    v_window record;
    v_reason text := 'ok';
    v_retry integer := NULL;
BEGIN
    -- A stale repeatable-read snapshot could survive waiting for the row lock.
    -- PostgREST's normal read-committed mode takes fresh snapshots after locking.
    IF pg_catalog.current_setting('transaction_isolation') <> 'read committed' THEN
        RETURN pg_catalog.jsonb_build_object('version', 1, 'ledger_id', p_ledger_id,
            'allowed', false, 'reason', 'storage', 'retry_after', NULL);
    END IF;
    SELECT * INTO v_metadata FROM housing_safety.metadata WHERE id = 1 FOR UPDATE;
    IF NOT FOUND OR v_metadata.version <> 1 OR p_ledger_id IS NULL
       OR v_metadata.ledger_id <> p_ledger_id THEN
        v_reason := 'storage';
    ELSIF p_visitor IS NULL OR p_visitor !~ '^[0-9a-f]{64}$'
       OR p_visitor_per_minute IS NULL OR p_visitor_per_minute < 1
       OR p_visitor_per_day IS NULL OR p_visitor_per_day < 1
       OR p_global_per_minute IS NULL OR p_global_per_minute < 1 THEN
        v_reason := 'settings';
    ELSE
        v_now := pg_catalog.clock_timestamp();
        FOR v_window IN
            SELECT * FROM (VALUES
                (60, p_visitor_per_minute, p_visitor),
                (86400, p_visitor_per_day, p_visitor),
                (60, p_global_per_minute, NULL::text)
            ) AS windows(seconds, request_limit, visitor)
        LOOP
            SELECT pg_catalog.count(*) INTO v_count FROM housing_safety.visitor_requests
            WHERE created_at > v_now - (v_window.seconds * interval '1 second')
              AND (v_window.visitor IS NULL OR visitor = v_window.visitor);
            IF v_count >= v_window.request_limit THEN
                SELECT created_at INTO v_release FROM housing_safety.visitor_requests
                WHERE created_at > v_now - (v_window.seconds * interval '1 second')
                  AND (v_window.visitor IS NULL OR visitor = v_window.visitor)
                ORDER BY created_at DESC OFFSET (v_window.request_limit - 1) LIMIT 1;
                v_retry := GREATEST(COALESCE(v_retry, 1), pg_catalog.ceil(
                    EXTRACT(epoch FROM (v_release + (v_window.seconds * interval '1 second') - v_now)))::integer);
                v_reason := 'rate';
            END IF;
        END LOOP;
        IF v_reason = 'ok' THEN
            -- Denied calls neither add records nor change the stored history.
            DELETE FROM housing_safety.visitor_requests WHERE created_at <= v_now - interval '24 hours';
            INSERT INTO housing_safety.visitor_requests (visitor, created_at) VALUES (p_visitor, v_now);
        END IF;
    END IF;
    RETURN pg_catalog.jsonb_build_object('version', 1, 'ledger_id', p_ledger_id,
        'allowed', v_reason = 'ok', 'reason', v_reason, 'retry_after', v_retry);
EXCEPTION WHEN OTHERS THEN
    RETURN pg_catalog.jsonb_build_object('version', 1, 'ledger_id', p_ledger_id,
        'allowed', false, 'reason', 'storage', 'retry_after', NULL);
END;
$function$;

CREATE FUNCTION public.housing_safety_status(p_ledger_id uuid)
RETURNS jsonb
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog
AS $function$
DECLARE
    v_metadata housing_safety.metadata%ROWTYPE;
    v_now timestamp with time zone;
    v_count32 bigint;
    v_count31 bigint;
BEGIN
    SELECT * INTO v_metadata FROM housing_safety.metadata WHERE id = 1;
    IF NOT FOUND OR v_metadata.version <> 1 OR p_ledger_id IS NULL
       OR v_metadata.ledger_id <> p_ledger_id THEN
        RETURN pg_catalog.jsonb_build_object('version', 1, 'ledger_id', p_ledger_id,
            'allowed', false, 'reason', 'storage', 'retry_after', NULL);
    END IF;
    v_now := pg_catalog.clock_timestamp();
    -- One snapshot for both counts; status does not admit or reserve anything.
    SELECT pg_catalog.count(*) FILTER (WHERE created_at > v_now - interval '768 hours'),
           pg_catalog.count(*) FILTER (WHERE created_at > v_now - interval '744 hours')
    INTO v_count32, v_count31 FROM housing_safety.rentcast_attempts;
    RETURN pg_catalog.jsonb_build_object('version', 1, 'ledger_id', p_ledger_id,
        'allowed', true, 'reason', 'ok', 'retry_after', NULL,
        'attempt_count32d', v_count32, 'attempt_count31d', v_count31);
EXCEPTION WHEN OTHERS THEN
    RETURN pg_catalog.jsonb_build_object('version', 1, 'ledger_id', p_ledger_id,
        'allowed', false, 'reason', 'storage', 'retry_after', NULL);
END;
$function$;

REVOKE ALL ON FUNCTION public.housing_safety_reserve(uuid, integer) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.housing_safety_admit(uuid, text, integer, integer, integer) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.housing_safety_status(uuid) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.housing_safety_reserve(uuid, integer) TO service_role;
GRANT EXECUTE ON FUNCTION public.housing_safety_admit(uuid, text, integer, integer, integer) TO service_role;
GRANT EXECUTE ON FUNCTION public.housing_safety_status(uuid) TO service_role;

SELECT ledger_id FROM housing_safety.metadata WHERE id = 1;
COMMIT;
