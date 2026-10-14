"""The "How Flip checks" page can't claim more than is true.

Every claim it shows comes from src/edgeorluck/data/claims.json. A claim stamped
"Proved" must name Lean theorems that exist in proofs/, the proof files must hold
no gaps (sorry, admit, native_decide, new axioms), and every Python function a
claim cites must exist. Lean itself checks the proofs (proofs/README.md); this
test keeps the page tied to them."""
import importlib
import json
import re
from pathlib import Path

import pytest

from edgeorluck.web.app import create_app
from edgeorluck.web.ledger import Ledger

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = json.loads((ROOT / "src" / "edgeorluck" / "data" / "claims.json").read_text(encoding="utf-8"))
PROOFS = ROOT / "proofs" / "EdgeOrLuck"


def _all_claims():
    for step in CLAIMS["steps"]:
        for claim in step["claims"]:
            yield step["id"], claim
    for claim in CLAIMS["limits"]:
        yield "limits", claim


def _code_of(path: Path) -> str:
    """The Lean source without comments and docstrings, which may mention the words checked for."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"/-.*?-/", "", text, flags=re.S)
    return re.sub(r"--[^\n]*", "", text)


def test_every_status_is_known_and_proved_claims_cite_proofs():
    for where, claim in _all_claims():
        assert claim["status"] in CLAIMS["statuses"], (where, claim["text"])
        if claim["status"] == "proved":
            assert claim.get("lean"), f"a 'Proved' claim with no proof: {claim['text']}"


@pytest.mark.parametrize("ref", sorted({r for _, c in _all_claims() for r in c.get("lean", [])}))
def test_cited_theorem_exists(ref):
    module, name = ref.split(".", 1)
    path = PROOFS / f"{module}.lean"
    assert path.exists(), f"no proof file {path.name} for {ref}"
    assert re.search(rf"^(theorem|lemma)\s+{re.escape(name)}\b", path.read_text(encoding="utf-8"), re.M), \
        f"{ref} is not a theorem in {path.name}"


@pytest.mark.parametrize("path", sorted(PROOFS.glob("*.lean")), ids=lambda p: p.name)
def test_proof_files_have_no_gaps(path):
    code = _code_of(path)
    for word in ("sorry", "admit", "native_decide"):
        assert not re.search(rf"\b{word}\b", code), f"{path.name} uses {word}"
    assert not re.search(r"^\s*axiom\s", code, re.M), f"{path.name} declares an axiom"
    assert "#print axioms" in code, f"{path.name} doesn't print the axioms its theorems use"


@pytest.mark.parametrize("ref", sorted({c["python"] for _, c in _all_claims() if c.get("python")}))
def test_cited_python_exists(ref):
    module, name = ref.rsplit(".", 1)
    assert hasattr(importlib.import_module(f"edgeorluck.{module}"), name), ref


def _share(r: float) -> str:
    v = r * 100
    return f"{v:.2f}".rstrip("0").rstrip(".") + "%" if v < 1 else f"{v:.1f}".rstrip("0").rstrip(".") + "%" \
        if v < 10 else f"{round(v)}%"


def test_measured_numbers_match_the_validation_run():
    """The page's measured numbers are the ones VALIDATION.md and validation.json report."""
    v = json.loads((ROOT / "src" / "edgeorluck" / "data" / "validation.json").read_text(encoding="utf-8"))
    worst = max([c["real_edge"]["rate"] for cells in v["false_real_edge"].values() for c in cells.values()]
                + [c["rate"] for c in v["checked_often"].values()])
    evidence = next(s for s in CLAIMS["steps"] if s["id"] == "evidence")
    measured = " ".join(c["text"] for c in evidence["claims"] if c["status"] == "measured")
    assert _share(v["headline"]["false_real_edge_typical"]) in measured and _share(worst) in measured
    versions = next(s for s in CLAIMS["steps"] if s["id"] == "versions")
    text = " ".join(c["text"] for c in versions["claims"] if c["status"] == "measured")
    assert _share(v["best_of_n"]["1000"]["without"]["rate"]) in text
    assert _share(v["best_of_n"]["1000"]["with"]["rate"]) in text
    anytime = next(s for s in CLAIMS["steps"] if s["id"] == "anytime")
    text = " ".join(c["text"] for c in anytime["claims"] if c["status"] == "measured")
    assert _share(v["checked_often"]["normal"]["rate"]) in text and _share(v["checked_often"]["autocorrelated"]["rate"]) in text


def test_the_page_and_its_data_are_served(tmp_path):
    client = create_app(Ledger(root=tmp_path)).test_client()
    page = client.get("/how")
    assert page.status_code == 200 and b"<html" in page.data
    assert client.get("/api/claims").get_json()["steps"][0]["id"] == CLAIMS["steps"][0]["id"]
    proof = client.get("/proofs/EdgeOrLuck/Mixture.lean")
    assert proof.status_code == 200 and b"theorem eMix_closed_form" in proof.data
    assert client.get("/proofs/../src/edgeorluck/judge.py").status_code == 404
    assert client.get("/proofs/EdgeOrLuck/nope.lean").status_code == 404
