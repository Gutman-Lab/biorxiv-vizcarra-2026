# Utilty functions.

import pixeltable as pxt
from pixeltable.exceptions import NotFoundError


def _ensure_schema_columns(table: pxt.Table, table_name: str, schema: dict) -> None:
    """Add any columns from schema that are missing on an existing table."""
    existing = set(table.columns())
    for col_name, col_type in schema.items():
        if col_name not in existing:
            print(f"Adding missing column {col_name!r} to {table_name!r}...")
            table.add_column(**{col_name: col_type}, if_exists="ignore")


def get_table_or_create(name: str, table_schema: dict) -> pxt.Table:
    """Get a table from Pixeltable. If the table does not exist, create it using the schema.

    Args:
        name: The name of the table to get or create.
        table_schema: The schema of the table to create.

    Returns:
        The table.

    Raises:
        Exception: If the table does not exist and cannot be created.

    """
    try:
        table = pxt.get_table(name)
        schema = table_schema.get("schema")
        if schema:
            _ensure_schema_columns(table, name, schema)
    except NotFoundError:
        # Table does not exist. Lets create it.
        print("Table not found, creating it...")

        schema = table_schema.get("schema")

        if schema is None:
            raise ValueError(f"Table schema must define a schema.")

        pk = table_schema.get("primary_key")

        if pk is None:
            raise ValueError(f"Table schema must define a primary key.")

        pk_cols = [pk] if isinstance(pk, str) else list(pk)

        for col in pk_cols:
            if col not in schema:
                raise ValueError(
                    f"Primary key column {col!r} is not in schema."
                )

        path = name.replace(".", "/")
        if "/" in path:
            parent = path.rsplit("/", 1)[0]
            pxt.create_dir(parent, if_exists="ignore", parents=True)

        table = pxt.create_table(
            path,
            schema=schema,
            primary_key=pk,
            comment=table_schema.get("description", ""),
        )

    return table
