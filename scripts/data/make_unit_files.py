#!/usr/bin/env python3
"""Write one mmCIF per gRNAde design unit, downloading whole entries as needed.

The pipeline addresses targets by design unit (``4H5O_1_L``), not by whole PDB
entry, so every target needs its own unit file.  This produces them from the
whole-entry mmCIFs, fetching anything missing from RCSB.


"""

from __future__ import annotations

import argparse
import logging
import sys
import zlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Bio.PDB import MMCIFParser, PDBIO, Select  # noqa: E402
from Bio.PDB.mmcifio import MMCIFIO  # noqa: E402

from evo_rna.config import PATHS  # noqa: E402
from evo_rna.data.download import download  # noqa: E402
from evo_rna.data.splits import StructureID, load_grnade  # noqa: E402
from evo_rna.profile.utils import RESNAME_TO_NUC, extract_chain_unit  # noqa: E402

log = logging.getLogger("make_unit_files")


# PDB format has a one-character chain field, so a unit spanning a chain like
# "WW" cannot be written as PDB without relabelling.  That is fine for a TEMPLATE
# LIBRARY -- RNAalign only needs coordinates for a TM-score -- but it is why the
# canonical unit files are mmCIF and the PDB copies live somewhere else.
_PDB_CHAIN_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


class _UnitSelect(Select):
    """Keep model 0 and the unit's chains.

    ``nucleotides_only`` restricts to residues :func:`extract_chain_unit` would
    keep -- those with a known RNA base -- dropping ions, ligands and modified
    bases.  It matters for files THIRD-PARTY tools read.  Our own readers filter
    by resname anyway, so an extra Mg or ligand is invisible to them (and to the
    sequence check this script verifies against); gRNAde parses the file itself,
    derives a sequence that includes the extras, and then fails its own
    assertion that the sequence and the predicted secondary structure are the
    same length.
    """

    def __init__(self, chains: tuple[str, ...], nucleotides_only: bool = False) -> None:
        self.chains = set(chains)
        self.nucleotides_only = nucleotides_only

    def accept_model(self, model):      # noqa: D102
        return model.id == 0

    def accept_chain(self, chain):      # noqa: D102
        return chain.id in self.chains

    def accept_residue(self, residue):  # noqa: D102
        if residue.id[0] == "W":        # het flag 'W' = water
            return False
        if self.nucleotides_only:
            return residue.get_resname().strip() in RESNAME_TO_NUC
        return True


def whole_entry_path(pdb_id: str, dirs: list[Path]) -> Path | None:
    for d in dirs:
        for name in (f"{pdb_id}.cif", f"{pdb_id}.pdb",
                     f"{pdb_id.lower()}.cif", f"{pdb_id.lower()}.pdb"):
            p = d / name
            if p.exists():
                return p
    return None


def write_unit(sid: StructureID, entry: Path, out: Path) -> tuple[bool, str]:
    """Write the unit file and verify it reads back as the whole entry does."""
    try:
        want_seq, _, _, want_keys = extract_chain_unit(str(entry), sid.chains)
    except Exception as exc:                                  # noqa: BLE001
        return False, f"cannot read {entry.name} chains {sid.chains}: {exc}"
    if not want_seq:
        return False, f"no RNA residues in {entry.name} chains {sid.chains}"

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(sid.pdb_id, str(entry))
    present = {c.id for c in structure[0].get_chains()}
    missing = set(sid.chains) - present
    if missing:
        return False, f"{entry.name} has no chain(s) {sorted(missing)}"

    # The temp name MUST keep a .cif suffix: extract_chain_unit dispatches its
    # parser on the extension, so a ".tmp" file is read as PDB and comes back
    # empty.  Write-then-rename still gives atomicity.
    tmp = out.parent / f".{out.stem}.tmp.cif"
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(str(tmp), _UnitSelect(sid.chains))

    # The verification that keeps this from becoming a third disagreeing source.
    try:
        got_seq, _, _, got_keys = extract_chain_unit(str(tmp), sid.chains)
    except Exception as exc:                                  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return False, f"unit file unreadable after write: {exc}"
    if got_seq != want_seq:
        tmp.unlink(missing_ok=True)
        return False, (f"MISMATCH: unit file gives {len(got_seq)} residues, "
                       f"whole entry gives {len(want_seq)}")
    if got_keys != want_keys:
        tmp.unlink(missing_ok=True)
        return False, "MISMATCH: chain/resnum keys differ from the whole entry"

    tmp.replace(out)
    return True, f"{len(want_seq)} residues, chains {'-'.join(sid.chains)}"


def write_pdb_template(sid: StructureID, entry: Path, out: Path,
                       n_expected: int) -> bool:
    """Write a PDB-format copy of the unit for the RNAalign template library.

    RNAalign is PDB-only -- it answers an mmCIF with "Cannot parse file. Chain
    number 0" and returns no alignment, so a library of mmCIF templates makes
    every PSSM flat and the evo term vanishes without an error anywhere.  The
    structsim split makes this total rather than partial: a train target's
    homologues are other TRAIN units, so if those are unreadable there is nothing
    left to align against.

    Chains are relabelled to single characters where they have to be; this copy
    is a search target, never the structure the pipeline reads.
    """
    try:
        structure = MMCIFParser(QUIET=True).get_structure(sid.pdb_id, str(entry))
        model = structure[0]
        for chain in list(model):
            if chain.id not in sid.chains:
                model.detach_child(chain.id)
        used = {c.id for c in model if len(c.id) == 1}
        for chain in model:
            if len(chain.id) != 1:
                chain.id = next(c for c in _PDB_CHAIN_POOL if c not in used)
                used.add(chain.id)
        tmp = out.parent / f".{out.stem}.tmp.pdb"
        io = PDBIO()
        io.set_structure(structure)
        # Nucleotides only: this copy is read by gRNAde / RIdiffusion / RDesign
        # and RNAalign, none of which share our resname filter.
        io.save(str(tmp), _UnitSelect(tuple(c.id for c in model),
                                      nucleotides_only=True))

        # The copy must hold exactly the residues the unit's sequence describes.
        # A tool that derives its own sequence from this file (gRNAde does) gets
        # a different length otherwise, and fails somewhere far from the cause --
        # in gRNAde's case an assertion that its sequence and its predicted
        # secondary structure are the same length.
        n_res = len({(l[21], l[22:27]) for l in tmp.read_text().splitlines()
                     if l.startswith(("ATOM", "HETATM"))})
        if n_res != n_expected:
            tmp.unlink(missing_ok=True)
            log.warning("[%s] PDB copy has %d residues, unit sequence has %d — "
                        "not written", sid, n_res, n_expected)
            return False
        tmp.replace(out)
        return True
    except Exception as exc:                                  # noqa: BLE001
        log.warning("[%s] no PDB template copy: %s", sid, exc)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="grnade_structsim")
    ap.add_argument("--subset", default="train", choices=("train", "val", "test"))
    ap.add_argument("--source-dir", type=Path, default=Path.home() / "database",
                    help="Directory of whole-entry mmCIFs to read from.")
    ap.add_argument("--out-dir", type=Path, default=Path(PATHS.pdb_dir),
                    help="Where the per-unit files go (default: configured pdb_dir).")
    ap.add_argument("--download-dir", type=Path, default=None,
                    help="Where fetched entries are cached (default: --source-dir).")
    ap.add_argument("--no-download", action="store_true",
                    help="Never fetch from RCSB; only use entries already present.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Regenerate unit files that already exist. Use this to "
                         "make the directory one consistent vintage.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--download-only", action="store_true",
                    help="Fetch missing whole entries and stop. Run this ONCE, "
                         "serially, before any array: sharding the download "
                         "hammers RCSB and races on the same entry file.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1,
                    help="Shard by PDB ENTRY, so units of one entry stay in one "
                         "task: the entry is parsed once per shard, and two "
                         "tasks never read a file the other might be writing.")
    ap.add_argument("--pdb-lib-dir", type=Path, default=None,
                    help="Also write a PDB-format copy of each unit here, for the "
                         "RNAalign template library (paths.pdb_lib). REQUIRED for "
                         "PSSM generation: RNAalign cannot parse mmCIF, so a "
                         "library of .cif templates yields flat PSSMs and a "
                         "silently absent evo term.")
    ap.add_argument("--archive-shadowed", action="store_true",
                    help="Move pre-existing <unit>.pdb files that would SHADOW "
                         "the .cif written here into <out-dir>/_superseded_pdb/. "
                         "unit lookup tries .pdb before .cif, so "
                         "without this the old file silently wins.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")

    sids = load_grnade(args.split)[args.subset]
    if args.limit:
        sids = sids[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dl_dir = args.download_dir or args.source_dir
    search = [args.source_dir, dl_dir, args.out_dir]

    if args.n_shards > 1:
        sids = [s for s in sids
                if (zlib.crc32(s.pdb_id.encode()) & 0xFFFFFFFF) % args.n_shards
                == args.shard]
        log.info("shard %d/%d: %d units, %d entries", args.shard, args.n_shards,
                 len(sids), len({s.pdb_id for s in sids}))
    # Track the two outputs SEPARATELY. They are written to different
    # directories for different consumers, and a unit can legitimately need one
    # and not the other -- which is exactly what happened when the template
    # library was added after the .cif files already existed: every unit was
    # "already present", so the library came out empty.
    def needs_cif(sid: StructureID) -> bool:
        # Either extension counts as present: unit lookup resolves .pdb or .cif.
        # (A pre-existing .pdb is present and OLD -- see the shadowing note.)
        return args.overwrite or not (
            (args.out_dir / f"{sid}.cif").exists()
            or (args.out_dir / f"{sid}.pdb").exists())

    def needs_pdb_template(sid: StructureID) -> bool:
        if args.pdb_lib_dir is None:
            return False
        return args.overwrite or not (args.pdb_lib_dir / f"{sid}.pdb").exists()

    todo = [s for s in sids if needs_cif(s) or needs_pdb_template(s)]
    n_cif_todo = sum(1 for s in todo if needs_cif(s))
    n_pdb_todo = sum(1 for s in todo if needs_pdb_template(s))
    log.info("%d units in %s:%s — %d need work (%d unit files, %d template "
             "copies)", len(sids), args.split, args.subset, len(todo),
             n_cif_todo, n_pdb_todo)

    need = {s.pdb_id for s in todo if whole_entry_path(s.pdb_id, search) is None}
    if need:
        if args.no_download or args.dry_run:
            log.info("%d entries not found locally%s: %s%s", len(need),
                     " (would download)" if args.dry_run else " and --no-download set",
                     ", ".join(sorted(need)[:8]), " ..." if len(need) > 8 else "")
        else:
            log.info("fetching %d missing entries from RCSB -> %s", len(need), dl_dir)
            download(need, dl_dir)

    if args.download_only:
        still = {s.pdb_id for s in todo if whole_entry_path(s.pdb_id, search) is None}
        log.info("download pass done; %d entries still unavailable", len(still))
        if still:
            print("unavailable entries: " + ", ".join(sorted(still)))
        return 0

    if args.dry_run:
        log.info("dry run: would write %d unit files to %s", len(todo), args.out_dir)
        return 0

    # A pre-existing <unit>.pdb SHADOWS the .cif written here, because
    # Unit lookup tries ".pdb" before ".cif".  Those older files are
    # a different vintage -- 2A2E_1_A reads 282 residues as one and 304 via
    # whole-entry chain selection -- so leaving them in place would mean this
    # entire pass had no effect for exactly the targets most likely to disagree.
    shadowed = [s for s in todo if (args.out_dir / f"{s}.pdb").exists()]

    ok = 0
    n_pdb = 0
    failures: list[str] = []
    for n, sid in enumerate(todo, 1):
        entry = whole_entry_path(sid.pdb_id, search)
        if entry is None:
            failures.append(f"{sid}: no entry file for {sid.pdb_id}")
            continue
        if needs_cif(sid):
            good, msg = write_unit(sid, entry, args.out_dir / f"{sid}.cif")
            if good:
                ok += 1
            else:
                failures.append(f"{sid}: {msg}")
        # Independent of the .cif: the template library is a separate output for
        # a separate consumer (RNAalign), and it must be fillable on its own.
        if needs_pdb_template(sid):
            args.pdb_lib_dir.mkdir(parents=True, exist_ok=True)
            try:
                want, _, _, _ = extract_chain_unit(str(entry), sid.chains)
            except Exception as exc:                          # noqa: BLE001
                failures.append(f"{sid}: template copy skipped ({exc})")
                continue
            n_pdb += write_pdb_template(sid, entry,
                                        args.pdb_lib_dir / f"{sid}.pdb",
                                        len(want))
        if n % 200 == 0:
            log.info("  %d/%d  (%d units, %d templates, %d failed)",
                     n, len(todo), ok, n_pdb, len(failures))

    if shadowed:
        archive = args.out_dir / "_superseded_pdb"
        if args.archive_shadowed:
            archive.mkdir(parents=True, exist_ok=True)
            for sid in shadowed:
                (args.out_dir / f"{sid}.pdb").rename(archive / f"{sid}.pdb")
            log.info("moved %d shadowing .pdb files to %s", len(shadowed), archive)
        else:
            log.warning(
                "%d units already have a .pdb here, which unit lookup resolves "
                "BEFORE the .cif written now — so those targets would keep using "
                "the old file. Re-run with --archive-shadowed to set them aside.",
                len(shadowed))

    print(f"\nwrote {ok} unit files to {args.out_dir}")
    if args.pdb_lib_dir is not None:
        print(f"  {n_pdb} PDB template copies -> {args.pdb_lib_dir}  "
              f"(point paths.pdb_lib / PDB_LIB here for PSSM generation)")
    mism = [f for f in failures if "MISMATCH" in f]
    if mism:
        print(f"  {len(mism)} DISCARDED for disagreeing with the whole entry — "
              f"these are the cases that would have made this a third structure "
              f"source; investigate before using them:")
        for f in mism[:10]:
            print(f"    {f}")
    other = [f for f in failures if "MISMATCH" not in f]
    if other:
        print(f"  {len(other)} could not be built:")
        for f in other[:10]:
            print(f"    {f}")
        if len(other) > 10:
            print(f"    ... and {len(other) - 10} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
