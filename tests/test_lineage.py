"""The provenance and licence gate rejects what it must, offline."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_lineage", ROOT / "scripts" / "check_lineage.py")
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)


def test_repo_lineage_passes_offline():
    lineage = cl.parse_lineage(ROOT / "lineage.yaml")
    gated = [e for k in ("models", "datasets") for e in lineage[k] if e.get("use") in cl.GATED_USES]
    assert gated, "lineage.yaml should list the training sources"
    assert not [p for e in lineage["models"] if e.get("use") in cl.GATED_USES for p in cl.model_problems(e, True)]
    assert not [p for e in lineage["datasets"] if e.get("use") in cl.GATED_USES for p in cl.dataset_problems(e)]


def test_unapproved_model_org_fails():
    e = {"name": "x", "hub_id": "someorg/some-model", "use": "teacher", "license": "Apache-2.0"}
    assert any("approved" in p for p in cl.model_problems(e, offline=True))


def test_untraceable_model_fails():
    assert cl.model_problems({"name": "x", "use": "train", "license": "MIT"}, offline=True)


def test_restrictive_licences_fail():
    for licence in ("CC-BY-NC-4.0", "unknown", "proprietary", ""):
        assert cl.dataset_problems({"name": "d", "use": "train", "license": licence}), licence


def test_licence_spellings_normalise():
    for licence in ("CC BY 4.0", "Apache 2.0", "CC0: Public Domain", "cc-by-sa-4.0", "MIT"):
        assert cl.permissive(licence), licence
