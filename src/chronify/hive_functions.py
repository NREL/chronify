import atexit
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from sqlalchemy import Engine, MetaData, text

from chronify.utils.path_utils import delete_if_exists


def create_materialized_view(
    query: str,
    dst_table: str,
    engine: Engine,
    metadata: MetaData,
    scratch_dir: Optional[Path] = None,
) -> None:
    """Create a materialized view with a Parquet file. This is a workaround for an undiagnosed
    problem with timestamps and time zones with hive.
    """
    with NamedTemporaryFile(dir=scratch_dir, suffix=".parquet") as f:
        f.close()
        output = Path(f.name)
    write_query = f"""
        INSERT OVERWRITE DIRECTORY
            '{output}'
            USING parquet
            ({query})
    """
    with engine.begin() as conn:
        conn.execute(text(write_query))
        atexit.register(lambda: delete_if_exists(output))
        view_query = f"CREATE VIEW {dst_table} AS SELECT * FROM parquet.`{output}`"
        conn.execute(text(view_query))
    metadata.reflect(engine, views=True)
