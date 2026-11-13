"""A MongoDB database that can only be read, enforced in code.

The legend selection report is meant to be run by the owner against the
production database, and the evergreen and child-bio scripts read production
in their dry runs. Whether they can write is not left to care:

- every collection handed out is a ReadOnlyCollection, which offers find,
  find_one, count_documents, estimated_document_count, distinct and aggregate
  and nothing else; any other attribute (insert, update, delete, replace,
  bulk_write, create_index, drop, rename, watch...) raises ReadOnlyViolation
  before a command is built;
- an aggregation is refused when any stage, at any depth, is $out or $merge;
- the database hands out nothing but those collections (no command, no
  create_collection, no drop_collection).

A database user with read-only rights makes this hold at the server too.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

READ_METHODS = frozenset({
    "find",
    "find_one",
    "count_documents",
    "estimated_document_count",
    "distinct",
    "aggregate",
})
WRITE_STAGES = ("$out", "$merge")


class ReadOnlyViolation(RuntimeError):
    """Something tried to write through a read-only database."""


def write_stage_in(value: Any) -> Optional[str]:
    """The first $out or $merge anywhere in a pipeline, nested stages included."""
    if isinstance(value, (list, tuple)):
        for item in value:
            found = write_stage_in(item)
            if found:
                return found
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            if key in WRITE_STAGES:
                return key
            found = write_stage_in(item)
            if found:
                return found
    return None


class ReadOnlyCollection:
    """A collection that can only be read. Asking for anything else is a loud
    error naming what was asked for, not a quiet attribute."""

    __slots__ = ("_collection", "_label")

    def __init__(self, collection: Any, label: str) -> None:
        object.__setattr__(self, "_collection", collection)
        object.__setattr__(self, "_label", label)

    @property
    def name(self) -> str:
        return self._collection.name

    def aggregate(self, pipeline: Iterable[Dict[str, Any]], *args: Any, **kwargs: Any) -> Any:
        stages = list(pipeline)
        stage = write_stage_in(stages)
        if stage:
            raise ReadOnlyViolation(f"{self._label} is read-only: {stage} refused on {self._collection.name}")
        return self._collection.aggregate(stages, *args, **kwargs)

    def __getattr__(self, attribute: str) -> Any:
        if attribute in READ_METHODS:
            return getattr(self._collection, attribute)
        raise ReadOnlyViolation(f"{self._label} is read-only: {attribute} is not allowed on {self._collection.name}")

    def __setattr__(self, attribute: str, value: Any) -> None:
        raise ReadOnlyViolation(f"{self._label} is read-only: cannot set {attribute}")


class ReadOnlyDatabase:
    """A database that hands out only read-only collections."""

    __slots__ = ("_db", "_label", "_collections")

    def __init__(self, db: Any, label: str) -> None:
        object.__setattr__(self, "_db", db)
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_collections", {})

    @property
    def name(self) -> str:
        return self._db.name

    def __getitem__(self, name: str) -> ReadOnlyCollection:
        found = self._collections.get(name)
        if found is None:
            found = ReadOnlyCollection(self._db[name], self._label)
            self._collections[name] = found
        return found

    def get_collection(self, name: str, *_args: Any, **_kwargs: Any) -> ReadOnlyCollection:
        return self[name]

    def list_collection_names(self, *args: Any, **kwargs: Any) -> Any:
        return self._db.list_collection_names(*args, **kwargs)

    def __getattr__(self, attribute: str) -> Any:
        raise ReadOnlyViolation(f"{self._label} is read-only: db.{attribute} is not allowed")

    def __setattr__(self, attribute: str, value: Any) -> None:
        raise ReadOnlyViolation(f"{self._label} is read-only: cannot set db.{attribute}")


def connect_read_only(uri: str, *, label: str, db_name: Optional[str] = None, app_name: str = "legend-read-only"):
    """(client, read-only database) for a URI. Reads from a secondary where
    there is one, to keep the load off the primary (that does not stop writes;
    the wrapper does). The database is the one named, else the one the
    pipeline would use (config.mongodb_utils.resolve_mongo_db_name)."""
    from pymongo import MongoClient

    from config.mongodb_utils import resolve_mongo_db_name

    client = MongoClient(
        uri,
        readPreference="secondaryPreferred",
        appname=app_name,
        serverSelectionTimeoutMS=30_000,
        retryWrites=False,
    )
    name = db_name or resolve_mongo_db_name(uri, client=client)
    return client, ReadOnlyDatabase(client[name], label)
