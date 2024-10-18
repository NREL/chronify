# Copied this code from https://github.com/sqlalchemy/sqlalchemy/wiki/Views

import sqlalchemy as sa
from sqlalchemy import Engine, MetaData, Selectable, TableClause
from sqlalchemy.ext import compiler
from sqlalchemy.schema import DDLElement
from sqlalchemy.sql import table


class CreateView(DDLElement):
    def __init__(self, name, selectable):
        self.name = name
        self.selectable = selectable


class DropView(DDLElement):
    def __init__(self, name):
        self.name = name


@compiler.compiles(CreateView)
def _create_view(element, compiler, **kw):
    return "CREATE VIEW %s AS %s" % (
        element.name,
        compiler.sql_compiler.process(element.selectable, literal_binds=True),
    )


@compiler.compiles(DropView)
def _drop_view(element, compiler, **kw):
    return "DROP VIEW %s" % (element.name)


def _view_exists(ddl, target, connection, **kw):
    return ddl.name in sa.inspect(connection).get_view_names()


def _view_doesnt_exist(ddl, target, connection, **kw):
    return not _view_exists(ddl, target, connection, **kw)


def create_view(
    name: str, selectable: Selectable, engine: Engine, metadata: MetaData
) -> TableClause:
    """Create a view from a selectable."""
    view = table(name)
    view._columns._populate_separate_keys(
        col._make_proxy(view)
        for col in selectable.selected_columns  # type: ignore
    )
    sa.event.listen(
        metadata,
        "after_create",
        CreateView(name, selectable).execute_if(callable_=_view_doesnt_exist),  # type: ignore
    )
    sa.event.listen(metadata, "before_drop", DropView(name).execute_if(callable_=_view_exists))  # type: ignore
    metadata.create_all(engine)
    return view