"""Plugin package."""

from __future__ import annotations

from recql.plugins.base import PluginRegistry, Retriever, Scorer, Reorderer, KvStore
from recql.plugins.composite import (
    CompositePluginRegistry,
    RoutedRetriever,
    RoutedSimilarityRetriever,
    RoutedTextSearchRetriever,
    RoutedScorer,
    RoutedFilterPlugin,
)
from recql.plugins.dialect import (
    SqlDialect,
    canonical_backend_name,
    compile_named_binds,
    get_sql_dialect,
    load_default_queries,
    normalize_backend_name,
)

__all__ = [
    "CompositePluginRegistry",
    "RoutedRetriever",
    "RoutedSimilarityRetriever",
    "RoutedTextSearchRetriever",
    "RoutedScorer",
    "RoutedFilterPlugin",
    "PluginRegistry",
    "Retriever",
    "Scorer",
    "Reorderer",
    "KvStore",
    "SqlDialect",
    "canonical_backend_name",
    "compile_named_binds",
    "get_sql_dialect",
    "load_default_queries",
    "normalize_backend_name",
]
