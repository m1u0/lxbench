import json
from pathlib import Path


FNV_OFFSET_BASIS = 2166136261
FNV_PRIME = 16777619
MANIFEST_FIELDS = {
    "version",
    "population_total",
    "sample_size",
    "seed",
    "selected_ids",
}


def _reject_json_constant(value):
    raise ValueError(f"non-JSON constant {value}")


def _sample_hash(value):
    result = FNV_OFFSET_BASIS
    for byte in value.encode("ascii"):
        result = ((result ^ byte) * FNV_PRIME) & 0xFFFFFFFF
    return result


def _select_ids(ids, sample_size, seed):
    ranked = sorted(
        (_sample_hash(f"{seed}:{prepared_id}"), prepared_id, index)
        for index, prepared_id in enumerate(ids)
    )
    selected_indices = {
        index for _, _, index in ranked[: min(sample_size, len(ranked))]
    }
    return [
        prepared_id
        for index, prepared_id in enumerate(ids)
        if index in selected_indices
    ]


def _manifest_path(output_path):
    return Path(f"{output_path}.manifest.json")


def _build_manifest(ids, sample_size, seed):
    if type(sample_size) is not int or sample_size < 1:
        raise ValueError("sample size must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return {
        "version": 1,
        "population_total": len(ids),
        "sample_size": sample_size,
        "seed": seed,
        "selected_ids": _select_ids(ids, sample_size, seed),
    }


def _load_manifest(output_path, ids):
    path = _manifest_path(output_path)
    if not path.exists():
        return None
    try:
        manifest = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except json.JSONDecodeError as error:
        raise ValueError("sample manifest is not valid JSON") from error
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS:
        raise ValueError("sample manifest has invalid fields")
    if type(manifest["version"]) is not int or manifest["version"] != 1:
        raise ValueError("sample manifest has an unsupported version")
    if (
        type(manifest["population_total"]) is not int
        or manifest["population_total"] != len(ids)
    ):
        raise ValueError("sample manifest population does not match prepared cases")

    expected = _build_manifest(ids, manifest["sample_size"], manifest["seed"])
    if manifest != expected:
        raise ValueError("sample manifest selection does not match its parameters")
    return manifest


def prepare_sample(output_path, ids, sample_size, seed):
    path = _manifest_path(output_path)
    if sample_size is None:
        if path.exists():
            raise ValueError("sample manifest exists but --sample-size was not provided")
        return list(ids)

    expected = _build_manifest(ids, sample_size, seed)
    if path.exists():
        if _load_manifest(output_path, ids) != expected:
            raise ValueError("sample manifest does not match the requested sample")
    elif output_path.exists():
        raise ValueError("sampled output exists without a sample manifest")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(expected, separators=(",", ":")) + "\n", encoding="utf-8"
        )
    return list(expected["selected_ids"])


def load_sample(responses_path, ids):
    manifest = _load_manifest(responses_path, ids)
    if manifest is None:
        return list(ids), None
    return list(manifest["selected_ids"]), {
        "sampled": True,
        "population_total": manifest["population_total"],
        "sample_seed": manifest["seed"],
    }
