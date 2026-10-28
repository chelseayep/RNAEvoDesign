"""Download RNA structure files (mmCIF) from RCSB PDB.

Resume-aware, polite, with retries.

Usage:
    python -m evo_rna.data.download --out data/pdb --split grnade_structsim
    python -m evo_rna.data.download --out data/pdb --split grnade_das
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import requests

from evo_rna.data.splits import load_grnade, unique_pdb_ids

logger = logging.getLogger(__name__)

RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


def _fetch_cif(pdb_id: str, out_dir: Path, session: requests.Session, retries: int = 3) -> bool:
    """Download ``{pdb_id}.cif`` into *out_dir*. Returns True on success."""
    dest = out_dir / f"{pdb_id}.cif"
    if dest.exists():
        return True

    url = RCSB_CIF_URL.format(pdb_id=pdb_id)
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=120)
            if resp.status_code == 404:
                logger.warning("%s: not found on RCSB (404)", pdb_id)
                return False
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return True
        except requests.RequestException as exc:
            logger.warning("%s: attempt %d/%d failed — %s", pdb_id, attempt, retries, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)

    logger.error("%s: all %d download attempts failed", pdb_id, retries)
    return False


def download(pdb_ids: set[str], out_dir: Path, delay: float = 0.05) -> tuple[int, int]:
    """Download CIF files for *pdb_ids* into *out_dir*. Returns (n_success, n_failed)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = sorted(pid for pid in pdb_ids if not (out_dir / f"{pid}.cif").exists())
    already = len(pdb_ids) - len(missing)
    logger.info("Resume check: %d/%d already present, %d remaining", already, len(pdb_ids), len(missing))
    if not missing:
        return len(pdb_ids), 0

    success = failed = 0
    with requests.Session() as session:
        session.headers["User-Agent"] = "EvoDesign/0.1 (research)"
        for i, pdb_id in enumerate(missing, 1):
            ok = _fetch_cif(pdb_id, out_dir, session)
            success += ok
            failed += not ok
            if i % 50 == 0:
                logger.info("  %d/%d fetched", i, len(missing))
            time.sleep(delay)

    logger.info("Done: %d/%d available  (new=%d failed=%d)", already + success, len(pdb_ids), success, failed)
    return success, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RNA CIF files from RCSB")
    parser.add_argument("--out", required=True, help="Output directory for CIF files")
    parser.add_argument(
        "--split",
        default="grnade_structsim",
        choices=list(("grnade_das", "grnade_structsim")),
        help="Which gRNAde split to download structures for",
    )
    parser.add_argument("--delay", type=float, default=0.05, help="Seconds between requests")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    split = load_grnade(args.split)
    ids = unique_pdb_ids(split)
    logger.info("Split %r: %d unique PDB IDs", args.split, len(ids))
    download(ids, Path(args.out), delay=args.delay)


if __name__ == "__main__":
    main()
