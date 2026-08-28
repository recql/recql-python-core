"""Run awaitables concurrently with cancellation support.

Adapted from graphql-core ``graphql.pyutils.gather_with_cancel``
(Apache-2.0 / MIT dual licensed upstream). Copied into RecQL so the
runtime does not depend on the ``graphql`` package.
"""

from __future__ import annotations

from asyncio import Future, ensure_future, gather
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable

__all__ = ["gather_with_cancel"]


async def gather_with_cancel(*awaitables: Awaitable[Any]) -> list[Any]:
    """Run awaitables concurrently; cancel siblings on first exception."""
    futures: list[Future[Any]] = [ensure_future(aw) for aw in awaitables]
    try:
        return await gather(*futures)
    except Exception:
        for future in futures:
            if not future.done():
                future.cancel()
        await gather(*futures, return_exceptions=True)
        raise
