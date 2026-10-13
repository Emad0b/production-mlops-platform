import json, sqlite3, time, uuid
from contextlib import contextmanager
from .config import DATABASE_URL, RUNTIME

PG = bool(DATABASE_URL)


@contextmanager
def db():
    if PG:
        import psycopg

        conn = psycopg.connect(DATABASE_URL)
    else:
        conn = sqlite3.connect(RUNTIME / "operations.db", timeout=30)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def sql(conn, query, args=()):
    return conn.execute(query.replace("?", "%s") if PG else query, args)


def initialize():
    with db() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS models (id TEXT PRIMARY KEY, kind TEXT, dataset TEXT, run_id TEXT, registry_version TEXT, artifact TEXT, metrics TEXT, validation TEXT, threshold REAL, schema_version TEXT, created DOUBLE PRECISION)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS predictions (id TEXT PRIMARY KEY, model_id TEXT, features TEXT, risk REAL, predicted INTEGER, label INTEGER, latency REAL, created DOUBLE PRECISION)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS reports (id TEXT PRIMARY KEY, kind TEXT, payload TEXT, created DOUBLE PRECISION)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS audit (id TEXT PRIMARY KEY, action TEXT, payload TEXT, created DOUBLE PRECISION)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rollouts (id TEXT PRIMARY KEY,candidate TEXT,champion TEXT,status TEXT,fraction REAL,policy TEXT,decision TEXT,created DOUBLE PRECISION)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rollout_observations (id TEXT PRIMARY KEY,rollout TEXT,prediction_id TEXT,model_id TEXT,ok INTEGER,latency REAL,created DOUBLE PRECISION)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS proposals (id TEXT PRIMARY KEY,dataset TEXT,reason TEXT,status TEXT,models TEXT,created DOUBLE PRECISION)"
        )
        for key in ["champion", "previous", "approved_dataset", "active_rollout"]:
            sql(
                conn,
                "INSERT INTO state VALUES (?,?) ON CONFLICT (key) DO NOTHING",
                (key, ""),
            )


def state(key):
    with db() as conn:
        row = sql(conn, "SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return row[0] if row else ""


def models():
    with db() as conn:
        rows = conn.execute("SELECT * FROM models ORDER BY created DESC").fetchall()
    keys = [
        "id",
        "kind",
        "dataset",
        "run_id",
        "registry_version",
        "artifact",
        "metrics",
        "validation",
        "threshold",
        "schema_version",
        "created",
    ]
    result = []
    for row in rows:
        m = dict(zip(keys, row))
        m["metrics"] = json.loads(m["metrics"])
        m["validation"] = json.loads(m["validation"])
        m.pop("artifact")
        result.append(m)
    return result


def model(mid):
    with db() as conn:
        row = sql(conn, "SELECT * FROM models WHERE id=?", (mid,)).fetchone()
    if not row:
        raise ValueError("Model not found")
    m = dict(
        zip(
            [
                "id",
                "kind",
                "dataset",
                "run_id",
                "registry_version",
                "artifact",
                "metrics",
                "validation",
                "threshold",
                "schema_version",
                "created",
            ],
            row,
        )
    )
    m["metrics"] = json.loads(m["metrics"])
    m["validation"] = json.loads(m["validation"])
    return m


def audit(action, payload, conn=None):
    args = (str(uuid.uuid4()), action, json.dumps(payload), time.time())
    if conn is not None:
        sql(conn, "INSERT INTO audit VALUES (?,?,?,?)", args)
    else:
        with db() as connection:
            sql(connection, "INSERT INTO audit VALUES (?,?,?,?)", args)


initialize()
