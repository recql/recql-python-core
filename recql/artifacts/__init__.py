"""Online artifact version pins / compatibility (Part 8.2).

Training stamps versions offline (examples/generator). Online bind/scorer
only resolves pins and refuses incompatible feature_spec / dims.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def config_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ArtifactPin:
    kind: str  # model | embedding
    name: str
    version: str
    dims: int | None = None
    config_hash: str | None = None
    feature_spec: dict[str, Any] | None = None


def pins_from_deployment(deployment: dict[str, Any] | None) -> dict[str, str]:
    if not deployment:
        return {}
    out: dict[str, str] = {}
    default = deployment.get("artifact_version")
    models = deployment.get("model_versions") or {}
    if isinstance(models, dict):
        for k, v in models.items():
            out[str(k)] = str(v)
    embeddings = deployment.get("embedding_versions") or {}
    if isinstance(embeddings, dict):
        for k, v in embeddings.items():
            out[str(k)] = str(v)
    if default:
        out.setdefault("__default__", str(default))
    return out


def resolve_version(name: str, pins: dict[str, str], *, fallback: str = "v1") -> str:
    if name in pins:
        return pins[name]
    return pins.get("__default__", fallback)


def check_feature_spec_compat(
    expected: dict[str, Any] | None, actual: dict[str, Any] | None
) -> None:
    from recql.errors import BindError

    if not expected or not actual:
        return
    exp_feats = expected.get("features")
    act_feats = actual.get("features")
    if exp_feats is not None and act_feats is not None and list(exp_feats) != list(act_feats):
        raise BindError(
            f"feature_spec mismatch: expected {exp_feats!r}, got {act_feats!r}"
        )
    exp_dims = expected.get("dims")
    act_dims = actual.get("dims")
    if exp_dims is not None and act_dims is not None and int(exp_dims) != int(act_dims):
        raise BindError(f"dims mismatch: expected {exp_dims}, got {act_dims}")


def check_embedding_dims(catalog_dims: int | None, artifact_dims: int | None) -> None:
    from recql.errors import BindError

    if catalog_dims is None or artifact_dims is None:
        return
    if int(catalog_dims) != int(artifact_dims):
        raise BindError(
            f"embedding dims mismatch: catalog={catalog_dims} artifact={artifact_dims}"
        )


ARTIFACT_REGISTRY_SQL = """
CREATE TABLE IF NOT EXISTS artifact_registry (
  kind TEXT NOT NULL,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  dims INT,
  config_hash TEXT,
  feature_spec JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (kind, name, version)
);
"""


async def ensure_artifact_registry(conn) -> None:
    await conn.execute(ARTIFACT_REGISTRY_SQL)


async def register_artifact(conn, pin: ArtifactPin) -> None:
    await ensure_artifact_registry(conn)
    await conn.execute(
        """
        INSERT INTO artifact_registry
          (kind, name, version, dims, config_hash, feature_spec)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
        ON CONFLICT (kind, name, version) DO UPDATE SET
          dims = EXCLUDED.dims,
          config_hash = EXCLUDED.config_hash,
          feature_spec = EXCLUDED.feature_spec,
          created_at = now()
        """,
        pin.kind,
        pin.name,
        pin.version,
        pin.dims,
        pin.config_hash,
        json.dumps(pin.feature_spec) if pin.feature_spec is not None else None,
    )


async def load_artifact_pin(
    conn, *, kind: str, name: str, version: str
) -> ArtifactPin | None:
    await ensure_artifact_registry(conn)
    row = await conn.fetchrow(
        """
        SELECT kind, name, version, dims, config_hash, feature_spec
        FROM artifact_registry
        WHERE kind=$1 AND name=$2 AND version=$3
        """,
        kind,
        name,
        version,
    )
    if row is None:
        return None
    spec = row["feature_spec"]
    if isinstance(spec, str):
        spec = json.loads(spec)
    return ArtifactPin(
        kind=row["kind"],
        name=row["name"],
        version=row["version"],
        dims=row["dims"],
        config_hash=row["config_hash"],
        feature_spec=dict(spec) if spec else None,
    )
