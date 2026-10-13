"""Write one real API result per sample to docs/examples/<id>.json.

    python tools/write_examples.py

The page is built against these files: they are exactly what POST /api/judge
returns for {"sample": id}.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgeorluck import samples  # noqa: E402
from edgeorluck.web.app import create_app  # noqa: E402
from edgeorluck.web.ledger import Ledger  # noqa: E402


def main() -> int:
    out = ROOT / "docs" / "examples"
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        client = create_app(Ledger(Path(tmp))).test_client()
        for s in samples.SAMPLES:
            result = client.post("/api/judge", json={"sample": s["id"]}).get_json()
            path = out / f"{s['id']}.json"
            path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
            print(path.relative_to(ROOT), result["verdict"]["code"], "|", result["verdict"]["headline"])
        result = client.post("/api/judge", json={"sample": "lucky", "variants_tried": 200}).get_json()
        path = out / "lucky_tried_200.json"
        path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(path.relative_to(ROOT), result["verdict"]["code"], "|", result["verdict"]["headline"])
        bad = client.post("/api/judge", json={"csv_text": "a,b\n1,2\n"}).get_json()
        path = out / "need_input.json"
        path.write_text(json.dumps(bad, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(path.relative_to(ROOT), bad["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
