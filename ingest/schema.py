import duckdb

_TABLES = [
    """
    CREATE TABLE IF NOT EXISTS demand (
        timestamp   TIMESTAMPTZ NOT NULL,
        region      VARCHAR     NOT NULL,
        demand_mw   DOUBLE,
        PRIMARY KEY (timestamp, region)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS generation (
        timestamp     TIMESTAMPTZ NOT NULL,
        region        VARCHAR     NOT NULL,
        fuel_type     VARCHAR     NOT NULL,
        generation_mw DOUBLE,
        PRIMARY KEY (timestamp, region, fuel_type)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS weather (
        timestamp           TIMESTAMPTZ NOT NULL,
        latitude            DOUBLE      NOT NULL,
        longitude           DOUBLE      NOT NULL,
        temperature_c       DOUBLE,
        wind_speed_10m_ms   DOUBLE,
        shortwave_radiation DOUBLE,
        is_forecast         BOOLEAN     DEFAULT FALSE,
        PRIMARY KEY (timestamp, latitude, longitude)
    )
    """,
]


def init_db(path: str = "gridpulse.duckdb") -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(path)
    for ddl in _TABLES:
        conn.execute(ddl)
    return conn
