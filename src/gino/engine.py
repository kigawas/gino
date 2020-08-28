from __future__ import annotations

import asyncio
import warnings
from collections import deque
from contextvars import ContextVar
from copy import copy
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.engine import create_engine as _create_engine
from sqlalchemy.engine.util import _distill_params
from sqlalchemy.engine.base import OptionEngineMixin
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import ResourceClosedError
from sqlalchemy.ext.asyncio import (
    exc as async_exc,
    AsyncConnection,
    AsyncEngine,
    AsyncTransaction,
)
from sqlalchemy.ext.asyncio.base import StartableContext
from sqlalchemy.sql import ClauseElement, WARN_LINTING
from sqlalchemy.util.concurrency import greenlet_spawn


async def create_engine(url, *arg, **kw):
    if kw.get("server_side_cursors", False):
        raise async_exc.AsyncMethodRequired(
            "Can't set server_side_cursors for async engine globally; "
            "use the connection.stream() method for an async "
            "streaming result set"
        )
    kw["future"] = True

    isolation_level = kw.pop("isolation_level", None)
    kw["isolation_level"] = "AUTOCOMMIT"

    u = make_url(url)
    if u.drivername in {"postgresql", "postgres"}:
        u = copy(u)
        u.drivername = "postgresql+asyncpg"

    min_size = kw.pop("min_size", 1)
    sync_engine = _create_engine(u, *arg, **kw)

    if min_size > 0:
        fs = [greenlet_spawn(sync_engine.connect) for i in range(min_size)]
        fs = (await asyncio.wait(fs))[0]
        if isolation_level:
            await greenlet_spawn(
                sync_engine.dialect.set_isolation_level,
                (await list(fs)[0]).connection,
                isolation_level,
            )
        fs = [greenlet_spawn((await fut).close) for fut in fs]
        await asyncio.wait(fs)

    if isolation_level is None:
        isolation_level = getattr(
            sync_engine.dialect, "default_isolation_level", "READ COMMITTED"
        )
    sync_engine = sync_engine.execution_options(tx_isolation_level=isolation_level)

    return GinoEngine(sync_engine)


class GinoConnection(AsyncConnection):
    __slots__ = (
        "_started",
        "_prev",
        "_next",
        "_connect_timeout",
        "_lazy",
        "_lock",
    )
    _prev: Optional[GinoConnection]
    _next: Optional[GinoConnection]

    def __init__(
        self, sync_engine: Engine, connect_timeout, lazy,
    ):
        super().__init__(sync_engine)
        self._connect_timeout = connect_timeout
        self._lazy = lazy
        self._lock = self._prev = self._next = None

    def begin(self) -> GinoTransaction:
        return GinoTransaction(self)

    def begin_nested(self) -> GinoTransaction:
        return GinoTransaction(self, nested=True)

    async def start(self):
        if not self._lazy:
            try:
                await self._sync_connection()
            except Exception:
                await self.release()
                raise
        return self

    async def _sync_connection(self):
        if self._prev and not self._next:
            return await self._prev._sync_connection()
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if not self.sync_engine:
                raise ValueError("This connection is already released permanently.")
            if not self.sync_connection:
                self.sync_connection = await greenlet_spawn(self.sync_engine.connect)
            return self.sync_connection

    async def close(self):
        await self.release()

    @property
    def dbapi_connection(self):
        """
        The current underlying database connection instance, type depends on
        the dialect in use. May be ``None`` if self is a lazy connection.

        """
        if self.sync_connection:
            return self.sync_connection.connection.connection
        elif self._prev:
            return self._prev.dbapi_connection

    @property
    def raw_connection(self):
        """
        The current underlying database connection instance, type depends on
        the dialect in use. May be ``None`` if self is a lazy connection.

        """
        return self.dbapi_connection._connection

    async def get_raw_connection(self, *, timeout=None):
        """
        Get the underlying database connection, acquire one if none present.

        :param timeout: Seconds to wait for the underlying acquiring
        :return: Underlying database connection instance depending on the
                 dialect in use
        :raises: :class:`~asyncio.TimeoutError` if the acquiring timed out

        """
        await self._sync_connection()
        return self.raw_connection

    async def _execute_sa10(self, object_, multiparams, params):
        params_20style = _distill_params(self, multiparams, params)
        if isinstance(object_, str):
            return await self.exec_driver_sql(object_, params_20style)
        else:
            return await self.execute(object_, params_20style)

    async def release(self, *, permanent=True):
        """
        Returns the underlying database connection to its pool.

        If ``permanent=False``, this connection will be set in lazy mode with
        underlying database connection returned, the next query on this
        connection will cause a new database connection acquired. This is
        useful when this connection may still be useful again later, while some
        long-running I/O operations are about to take place, which should not
        take up one database connection or even transaction for that long time.

        Otherwise with ``permanent=True`` (default), this connection will be
        marked as closed after returning to pool, and be no longer usable
        again.

        If this connection is a reusing connection, then only this connection
        is closed (depending on ``permanent``), the reused underlying
        connection will **not** be returned back to the pool.

        Practically it is recommended to return connections in the reversed
        order as they are borrowed, but if this connection is a reused
        connection with still other opening connections reusing it, then on
        release the underlying connection **will be** returned to the pool,
        with all the reusing connections losing an available underlying
        connection. The availability of further operations on those reusing
        connections depends on the given ``permanent`` value.

        .. seealso::

            :meth:`.GinoEngine.acquire`

        """
        if permanent:
            self.sync_engine = None
            if self._next:
                self._next._prev = self._prev
                if self._prev:
                    self._prev._next = self._next

        if self.sync_connection:
            async with self._lock:
                conn, self.sync_connection = self.sync_connection, None
                if conn:
                    await greenlet_spawn(conn.close)
        # elif self._prev:
        #     await self._prev.release(permanent=permanent)

    async def all(self, clause, *multiparams, **params):
        """
        Runs the given query in database, returns all results as a list.

        This method accepts the same parameters taken by SQLAlchemy
        :meth:`~sqlalchemy.engine.Connectable.execute`. You can pass in a raw
        SQL string, or *any* SQLAlchemy query clauses.

        If the given query clause is built by CRUD models, then the returning
        rows will be turned into relevant model objects (Only one type of model
        per query is supported for now, no relationship support yet). See
        :meth:`execution_options` for more information.

        If the given parameters are parsed as "executemany" - bulk inserting
        multiple rows in one call for example, the returning result from
        database will be discarded and this method will return ``None``.

        """
        result = await self._execute_sa10(clause, multiparams, params)
        return result.all()

    async def first(self, clause, *multiparams, **params):
        """
        Runs the given query in database, returns the first result.

        If the query returns no result, this method will return ``None``.

        See :meth:`all` for common query comments.

        """
        result = await self._execute_sa10(clause, multiparams, params)
        try:
            return result.first()
        except ResourceClosedError as e:
            warnings.warn(
                "GINO 2.0 will raise ResourceClosedError: " + str(e),
                DeprecationWarning,
            )

    async def one_or_none(self, clause, *multiparams, **params):
        """
        Runs the given query in database, returns at most one result.

        If the query returns no result, this method will return ``None``.
        If the query returns multiple results, this method will raise
        :class:`~gino.exceptions.MultipleResultsFound`.

        See :meth:`all` for common query comments.

        """
        result = await self._execute_sa10(clause, multiparams, params)
        return result.one_or_none()

    async def one(self, clause, *multiparams, **params):
        """
        Runs the given query in database, returns exactly one result.

        If the query returns no result, this method will raise
        :class:`~gino.exceptions.NoResultFound`.
        If the query returns multiple results, this method will raise
        :class:`~gino.exceptions.MultipleResultsFound`.

        See :meth:`all` for common query comments.

        """
        result = await self._execute_sa10(clause, multiparams, params)
        return result.one()

    async def scalar(self, clause, *multiparams, **params):
        """
        Runs the given query in database, returns the first result.

        If the query returns no result, this method will return ``None``.

        See :meth:`all` for common query comments.

        """
        result = await self._execute_sa10(clause, multiparams, params)
        return result.scalar()

    async def status(self, clause, *multiparams, **params):
        """
        Runs the given query in database, returns the query status.

        The returning query status depends on underlying database and the
        dialect in use. For asyncpg it is a string, you can parse it like this:
        https://git.io/v7oze

        """
        result = await self._execute_sa10(clause, multiparams, params)
        return f"SELECT {result.rowcount}", result.all()

    class _IterateResult(StartableContext):
        def __init__(self, conn, *args):
            self._conn = conn
            self._result = None
            self._args = args

        async def start(self) -> "StartableContext":
            if not self._result:
                self._result = await self._conn.stream(*self._args)
            return self

        async def next(self):
            try:
                return await self._result.fetchone()
            except ResourceClosedError as e:
                warnings.warn(
                    "GINO 2.0 will raise ResourceClosedError: " + str(e),
                    DeprecationWarning,
                )

        async def __aexit__(self, type_, value, traceback):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            await self.start()
            return await self._result.__anext__()

        def __getattr__(self, item):
            return getattr(self._result, item)

    def iterate(self, clause, *multiparams, **params):
        """
        Creates a server-side cursor in database for large query results.

        Cursors must work within transactions::

            async with conn.transaction():
                async for user in conn.iterate(User.query):
                    # handle each user without loading all users into memory

        Alternatively, you can manually control how the cursor works::

            async with conn.transaction():
                cursor = await conn.iterate(User.query)
                user = await cursor.next()
                users = await cursor.many(10)

        Read more about how :class:`~gino.dialects.base.Cursor` works.

        Similarly, this method takes the same parameters as :meth:`all`.

        """
        params_20style = _distill_params(self, multiparams, params)
        if isinstance(clause, str):
            clause = text(clause)
        return self._IterateResult(self, clause, params_20style)


ASYNCPG_ISOLATION_MAPPING = dict(
    read_committed="READ COMMITTED",
    repeatable_read="REPEATABLE READ",
    serializable="SERIALIZABLE",
)


class GinoEngine(AsyncEngine):
    __slots__ = ("_hat",)

    _connection_cls = GinoConnection

    _option_cls: type

    class _gino_trans_ctx(AsyncEngine._trans_ctx):
        def __init__(self, conn, kwargs):
            super().__init__(conn)
            self._kwargs = kwargs

        async def start(self):
            await super().start()
            # TODO: different dialects?

            query = ""
            isolation = self._kwargs.get("isolation")
            current_isolation = self.conn.sync_connection.get_execution_options()[
                "tx_isolation_level"
            ]
            if isolation:
                isolation = ASYNCPG_ISOLATION_MAPPING.get(isolation)
                if isolation:
                    if isolation != current_isolation:
                        query += "ISOLATION LEVEL " + isolation
            if current_isolation == "SERIALIZABLE":
                if self._kwargs.get("readonly"):
                    query += " READ ONLY"
                if self._kwargs.get("deferrable"):
                    query += " DEFERRABLE"
            if query:
                await self.conn.exec_driver_sql(f"SET TRANSACTION {query};")

            return self.transaction

    def __init__(self, sync_engine: Engine):
        super().__init__(sync_engine)
        self._hat = ContextVar("hat")

    @property
    def dialect(self):
        """
        Read-only property for the
        :class:`~sqlalchemy.engine.interfaces.Dialect` of this engine.

        """
        return self.sync_engine.dialect

    def connect(self) -> GinoConnection:
        return self.acquire(timeout=None, reuse=False, lazy=False, reusable=False)

    def acquire(self, *, timeout=None, reuse=False, lazy=False, reusable=True):
        """
        reusable    reuse   head    push    reuse
        no          -       -       no      None
        yes         no      no      yes     None
        yes         no      yes     yes     None
        yes         yes     no      yes     None
        yes         yes     yes     no      head
        """
        rv = self._connection_cls(self.sync_engine, timeout, lazy)
        if reusable:
            try:
                hat = self._hat.get()
            except LookupError:
                hat = self._connection_cls(None, None, None)
                self._hat.set(hat)
            head = rv._prev = hat._prev
            if not (reuse and head):
                hat._prev = rv
                rv._next = hat
                if head:
                    head._next = rv
        return rv

    @property
    def current_connection(self):
        """
        Gets the most recently acquired reusable connection in the context.
        ``None`` if there is no such connection.

        :return: :class:`.GinoConnection`

        """
        hat = self._hat.get()
        if hat:
            return hat._prev

    async def close(self):
        """
        Close the engine, by closing the underlying pool.

        """
        await greenlet_spawn(self.sync_engine.dispose)

    async def all(self, clause, *multiparams, **params):
        """
        Acquires a connection with ``reuse=True`` and runs
        :meth:`~.GinoConnection.all` on it. ``reuse=True`` means you can safely
        do this without borrowing more than one underlying connection::

            async with engine.acquire():
                await engine.all('SELECT ...')

        The same applies for other query methods.

        """
        async with self.acquire(reuse=True) as conn:
            return await conn.all(clause, *multiparams, **params)

    async def first(self, clause, *multiparams, **params):
        """
        Runs :meth:`~.GinoConnection.first`, See :meth:`.all`.

        """
        async with self.acquire(reuse=True) as conn:
            return await conn.first(clause, *multiparams, **params)

    async def one_or_none(self, clause, *multiparams, **params):
        """
        Runs :meth:`~.GinoConnection.one_or_none`, See :meth:`.all`.

        """
        async with self.acquire(reuse=True) as conn:
            return await conn.one_or_none(clause, *multiparams, **params)

    async def one(self, clause, *multiparams, **params):
        """
        Runs :meth:`~.GinoConnection.one`, See :meth:`.all`.

        """
        async with self.acquire(reuse=True) as conn:
            return await conn.one(clause, *multiparams, **params)

    async def scalar(self, clause, *multiparams, **params):
        """
        Runs :meth:`~.GinoConnection.scalar`, See :meth:`.all`.

        """
        async with self.acquire(reuse=True) as conn:
            return await conn.scalar(clause, *multiparams, **params)

    async def status(self, clause, *multiparams, **params):
        """
        Runs :meth:`~.GinoConnection.status`. See also :meth:`.all`.

        """
        async with self.acquire(reuse=True) as conn:
            return await conn.status(clause, *multiparams, **params)

    class _CompileConnection:
        def __init__(self, dialect):
            self.dialect = dialect
            self._is_future = True

    class _CompileDBAPIConnection:
        def cursor(self):
            pass

    def compile(self, clause: ClauseElement, *multiparams, **params):
        """
        A shortcut for :meth:`~gino.dialects.base.AsyncDialectMixin.compile` on
        the dialect, returns raw SQL string and parameters according to the
        rules of the dialect.

        """
        distilled_params = _distill_params(self, multiparams, params)
        if distilled_params:
            # ensure we don't retain a link to the view object for keys()
            # which links to the values, which we don't want to cache
            keys = sorted(distilled_params[0])
            for_executemany = len(distilled_params) > 1
        else:
            keys = []
            for_executemany = False
        compiled_sql, extracted_params, cache_hit = clause._compile_w_cache(
            dialect=self.sync_engine.dialect,
            compiled_cache=self.sync_engine._compiled_cache,
            column_keys=keys,
            for_executemany=for_executemany,
            schema_translate_map=None,
            linting=self.sync_engine.dialect.compiler_linting | WARN_LINTING,
        )
        context = self.sync_engine.dialect.execution_ctx_cls._init_compiled(
            self.sync_engine.dialect,
            self._CompileConnection(self.sync_engine.dialect),
            self._CompileDBAPIConnection(),
            {},
            compiled_sql,
            distilled_params,
            clause,
            extracted_params,
            cache_hit=cache_hit,
        )
        parameters = context.parameters
        if not context.executemany:
            parameters = parameters[0]
        return context.statement, parameters

    def transaction(self, *, timeout=None, reuse=True, reusable=True, **kwargs):
        """
        Borrows a new connection and starts a transaction with it.

        Different to :meth:`.GinoConnection.transaction`, transaction on engine
        level supports only managed usage::

            async with engine.transaction() as tx:
                # play with transaction here

        Where the implicitly acquired connection is available as
        :attr:`tx.connection <gino.transaction.GinoTransaction.connection>`.

        By default, :meth:`.transaction` acquires connection with
        ``reuse=True`` and ``reusable=True``, that means it by default tries to
        create a nested transaction instead of a new transaction on a new
        connection. You can change the default behavior by setting these two
        arguments.

        The other arguments are the same as
        :meth:`~.GinoConnection.transaction` on connection.

        .. seealso::

            :meth:`.GinoEngine.acquire`

            :meth:`.GinoConnection.transaction`

            :class:`~gino.transaction.GinoTransaction`

        :return: A asynchronous context manager that yields a
          :class:`~gino.transaction.GinoTransaction`

        """
        conn = self.acquire(timeout=timeout, reuse=reuse, reusable=reusable)
        return self._gino_trans_ctx(conn, kwargs)

    def iterate(self, clause, *multiparams, **params):
        """
        Creates a server-side cursor in database for large query results.

        This requires that there is a reusable connection in the current
        context, and an active transaction is present. Then its
        :meth:`.GinoConnection.iterate` is executed and returned.

        """
        connection = self.current_connection
        if connection is None:
            raise ValueError("No Connection in context, please provide one")
        return connection.iterate(clause, *multiparams, **params)


class AsyncOptionEngine(OptionEngineMixin, GinoEngine):
    pass


GinoEngine._option_cls = AsyncOptionEngine


class GinoTransaction(AsyncTransaction):
    async def start(self):
        conn = await self.connection._sync_connection()
        in_trans = conn.in_transaction()
        if not in_trans:
            conn = conn.execution_options(
                isolation_level=conn.get_execution_options()["tx_isolation_level"]
            )
        elif not self.nested:
            self.nested = True

        self.sync_transaction = await greenlet_spawn(
            conn.begin_nested if self.nested else conn.begin
        )
        if not in_trans:
            if conn.dialect.driver == "asyncpg":
                await conn.connection.connection._start_transaction()
        return self
