#!/usr/bin/env python3
"""
Fetch RNA structures from RNAsolo API.

API template (from RNAsolo docs):
  https://rnasolo.cs.put.poznan.pl/api/query/structure
    ?pdbid=<PDB_ID>
    &format=pdb
    &chains=<chains>
    &models=<models>
    &rfam=<range>

Usage examples:
  # Mirror a previously-generated local manifest onto another machine/cluster
  python scripts/fetch_rnapdb.py \
    --manifest data/rnasolo/rnasolo_manifest.csv \
    --out-root data/rnasolo/local

  # Fetch a single structure (chain list and model optional; see --help)
  python scripts/fetch_rnapdb.py --pdbid 1A3M --chains A,B --models 1 --out one.pdb

  # Fetch from the flat manifest and write into a single directory
  python scripts/fetch_rnapdb.py \
    --manifest data/rnasolo/rnasolo_manifest_flat.csv \
    --out-root data/rnasolo/fetched_flat
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

try:
    import ssl
except Exception:  # pragma: no cover
    ssl = None

API_BASE = "https://rnasolo.cs.put.poznan.pl/api/query/structure"
API_PACKAGE = "https://rnasolo.cs.put.poznan.pl/api/query/structure/package"
USER_AGENT = "RNAEvoDesign/1.0"


@dataclass(frozen=True)
class FetchSpec:
    pdbid: str
    chains: str = ""  # comma-separated, optional
    models: str = ""  # comma-separated, optional
    rfam: str = ""  # optional range "start-end"
    out_path: Path = Path()


def _norm_csv_cell(x):
    if x is None:
        return ""
    return str(x).strip().strip('"')


def _build_url(pdbid: str, fmt: str, chains: str = "", models: str = "", rfam: str = "") -> str:
    params: Dict[str, str] = {"pdbid": pdbid, "format": fmt}
    if chains:
        params["chains"] = chains
    if models:
        params["models"] = models
    if rfam:
        params["rfam"] = rfam
    return API_BASE + "?" + urlencode(params)


def _build_package_url(pdbid: str, fmt: str, models: str = "") -> str:
    params: Dict[str, str] = {"pdbid": pdbid, "format": fmt}
    if models:
        params["models"] = models
    return API_PACKAGE + "?" + urlencode(params)


def _download(url: str, timeout_s: int = 60, insecure: bool = False) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    context = None
    if ssl is not None:
        try:
            if insecure and hasattr(ssl, "_create_unverified_context"):
                context = ssl._create_unverified_context()  # type: ignore[attr-defined]
            else:
                context = ssl.create_default_context()
        except Exception:
            context = None
    if context is not None:
        with urlopen(req, timeout=timeout_s, context=context) as resp:
            return resp.read()
    with urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def _looks_like_error_payload(data: bytes) -> bool:
    if not data:
        return True
    head = data[:200].lower()
    # RNAsolo sometimes returns JSON-ish error bodies; keep conservative.
    return (b"error" in head) or (b"not found" in head) or (b"invalid" in head)


def fetch_one(
    spec: FetchSpec, fmt: str, force: bool, sleep_s: float, retries: int, insecure: bool
) -> bool:
    if not force and spec.out_path.exists() and spec.out_path.stat().st_size > 0:
        return True

    spec.out_path.parent.mkdir(parents=True, exist_ok=True)

    url = _build_url(
        pdbid=spec.pdbid,
        fmt=fmt,
        chains=spec.chains,
        models=spec.models,
        rfam=spec.rfam,
    )
    package_url = _build_package_url(pdbid=spec.pdbid, fmt=fmt, models=spec.models)
    package_url_nomodels = _build_package_url(pdbid=spec.pdbid, fmt=fmt, models="")

    last_err = None  # type: Optional[BaseException]
    last_http_body = None  # type: Optional[bytes]
    for attempt in range(retries + 1):
        try:
            data = _download(url, insecure=insecure)
            if _looks_like_error_payload(data):
                raise ValueError("empty/error payload")
            with spec.out_path.open("wb") as f:
                f.write(data)
            if sleep_s:
                time.sleep(sleep_s)
            return True
        except HTTPError as e:
            last_err = e
            try:
                last_http_body = e.read()
            except Exception:
                last_http_body = None

            # Fallback: if chain-specific request returns "No result", try the package endpoint
            # (which returns a cleaned structure containing chains in RNAsolo2).
            if (
                int(getattr(e, "code", 0) or 0) == 400
                and spec.chains
                and last_http_body
                and b"No result" in last_http_body
            ):
                try:
                    sys.stderr.write(
                        "Fallback: pdbid={} chains={} models={} rfam={} -> trying package endpoint\n".format(
                            spec.pdbid,
                            spec.chains or "-",
                            spec.models or "-",
                            spec.rfam or "-",
                        )
                    )
                    # Try package with models first, then without models (some entries error when models is set).
                    for pkg_url in (package_url, package_url_nomodels):
                        try:
                            sys.stderr.write("  try package url: {}\n".format(pkg_url))
                            pkg_data = _download(pkg_url, insecure=insecure)
                            if _looks_like_error_payload(pkg_data):
                                continue
                            with spec.out_path.open("wb") as f:
                                f.write(pkg_data)
                            if sleep_s:
                                time.sleep(sleep_s)
                            sys.stderr.write(
                                "  OK via package fallback -> saved {}\n".format(spec.out_path)
                            )
                            return True
                        except HTTPError as pkg_http:
                            last_err = pkg_http
                            try:
                                last_http_body = pkg_http.read()
                            except Exception:
                                last_http_body = None
                        except Exception as pkg_e:
                            last_err = pkg_e
                            last_http_body = None
                except Exception as pkg_outer:
                    last_err = pkg_outer
                    last_http_body = None

            if attempt < retries:
                time.sleep(min(5.0, 0.5 * (attempt + 1)))
                continue
            break
        except (URLError, OSError, ValueError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(min(5.0, 0.5 * (attempt + 1)))
                continue
            break

    if last_http_body:
        body_preview = last_http_body[:200]
        sys.stderr.write(
            "FAILED pdbid={} chains={} models={} rfam={} -> {} ({}; url={}; body={})\n".format(
                spec.pdbid,
                spec.chains or "-",
                spec.models or "-",
                spec.rfam or "-",
                spec.out_path,
                last_err,
                url,
                body_preview,
            )
        )
    else:
        sys.stderr.write(
            "FAILED pdbid={} chains={} models={} rfam={} -> {} ({}; url={})\n".format(
                spec.pdbid,
                spec.chains or "-",
                spec.models or "-",
                spec.rfam or "-",
                spec.out_path,
                last_err,
                url,
            )
        )
    return False


def iter_specs_from_manifest(manifest_path: Path, out_root: Path) -> Iterable[FetchSpec]:
    with manifest_path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            pdbid = _norm_csv_cell(row.get("pdb_id") or row.get("pdbid")).upper()
            if not pdbid:
                continue

            # Prefer explicit "chains" column (comma-separated chain IDs).
            chains = _norm_csv_cell(row.get("chains"))
            models = _norm_csv_cell(row.get("model") or row.get("models"))
            rfam = _norm_csv_cell(row.get("rfam_range") or row.get("rfam"))

            out_name = _norm_csv_cell(row.get("out_name") or row.get("out"))
            if out_name:
                out_path = (out_root / out_name).resolve()
            else:
                relpath = _norm_csv_cell(row.get("relpath") or row.get("local_path"))
                if relpath and ("PDB_" in relpath or relpath.endswith(".pdb")):
                    # If relpath points to an existing nested file, preserve it under out_root.
                    out_path = (out_root / relpath).resolve()
                else:
                    out_path = (out_root / "{}.pdb".format(pdbid)).resolve()

            yield FetchSpec(pdbid=pdbid, chains=chains, models=models, rfam=rfam, out_path=out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch RNA PDB files from RNAsolo API.")
    ap.add_argument("--manifest", default=None, help="Manifest CSV from parse_rnasolo_local_manifest.py")
    ap.add_argument(
        "--out-root",
        default="data/rna_pdb",
        help="Root output directory when using --manifest (default: %(default)s)",
    )
    ap.add_argument("--pdbid", default=None, help="Single PDB id (e.g. 1A3M)")
    ap.add_argument("--chains", default="", help="Comma-separated chains (e.g. A,B). Optional.")
    ap.add_argument("--models", default="", help="Model id(s) separated by comma (e.g. 1 or 1,2). Optional.")
    ap.add_argument("--rfam", default="", help="Optional Rfam range like 254-323")
    ap.add_argument("--out", default=None, help="Output path for single fetch (default: <out-root>/<pdbid>.pdb)")
    ap.add_argument(
        "--format",
        default="pdb",
        help="RNAsolo format (default: pdb). Examples: pdb, mmCIF, fasta, bpseq, dbn, csv",
    )
    ap.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification (use only if your environment lacks CA certs).",
    )
    ap.add_argument("--force", action="store_true", help="Re-download even if file exists")
    ap.add_argument("--sleep", type=float, default=0.1, help="Sleep between downloads in seconds (default: %(default)s)")
    ap.add_argument("--retries", type=int, default=2, help="Retries per file on error (default: %(default)s)")
    ap.add_argument(
        "--max-n",
        type=int,
        default=0,
        help="If >0, download only the first N rows from --manifest (useful for testing).",
    )
    args = ap.parse_args()

    fmt = args.format
    force = bool(args.force)
    sleep_s = float(args.sleep)
    retries = int(args.retries)
    insecure = bool(args.insecure)

    ok = 0
    total = 0
    skipped_existing = 0

    if args.manifest:
        manifest_path = Path(os.path.expanduser(args.manifest)).resolve()
        out_root = Path(os.path.expanduser(args.out_root)).resolve()
        if not manifest_path.exists():
            sys.stderr.write("Manifest not found: {}\n".format(manifest_path))
            return 2
        for spec in iter_specs_from_manifest(manifest_path, out_root=out_root):
            if args.max_n and total >= int(args.max_n):
                break
            if not force and spec.out_path.exists() and spec.out_path.stat().st_size > 0:
                skipped_existing += 1
                total += 1
                continue
            total += 1
            if fetch_one(
                spec, fmt=fmt, force=force, sleep_s=sleep_s, retries=retries, insecure=insecure
            ):
                ok += 1
        print(
            "Done: {}/{} downloaded, {} skipped existing (out-root={})".format(
                ok, total, skipped_existing, out_root
            )
        )
        return 0 if ok == total else 1

    if args.pdbid:
        out_root = Path(os.path.expanduser(args.out_root)).resolve()
        out_path = Path(os.path.expanduser(args.out)).resolve() if args.out else (out_root / "{}.pdb".format(args.pdbid.upper())).resolve()
        spec = FetchSpec(
            pdbid=args.pdbid.upper(),
            chains=args.chains,
            models=args.models,
            rfam=args.rfam,
            out_path=out_path,
        )
        total = 1
        ok = (
            1
            if fetch_one(spec, fmt=fmt, force=force, sleep_s=0.0, retries=retries, insecure=insecure)
            else 0
        )
        if ok:
            print("Wrote: {}".format(out_path))
            return 0
        return 1

    sys.stderr.write("Error: provide --manifest PATH or --pdbid ID\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

