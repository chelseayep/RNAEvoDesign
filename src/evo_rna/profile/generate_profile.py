"""
Given input PDB, run RNAalign, optionally write msa.txt, build PSSM weights with mkprf.compute_w.

In-memory ``RnaAlignmentHit`` list feeds ``compute_w`` directly — no msa.txt round-trip required.
"""

import argparse
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


from evo_rna.profile.pssm import (
    RnaAlignmentHit,
    compute_M,
    write_msa_txt,
    write_prf,
    fetch_top_hits,
)
from evo_rna.profile.features import build_structure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _quick_residue_count(path: Path) -> int:
    """Cheap nucleotide count via the per-residue C1' atom (no full structure parse).

    There is exactly one ``C1'`` atom per nucleotide, in both PDB and mmCIF atom
    records, so counting ``C1'`` lines is an O(file) size estimate.  Used to skip
    oversized templates before invoking RNAalign, which fails to parse very large
    chains (rRNA / PDB atom-serial overflow → ``Chain number 0``).  Returns -1 if
    the file can't be read, so RNAalign still gets a chance at it.
    """
    try:
        n = 0
        with open(path, "r", errors="ignore") as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")) and "C1'" in line:
                    n += 1
        return n
    except OSError:
        return -1


def _fit_length(pssm: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or truncate an (L, 4) PSSM to ``target_length`` rows.

    Padding rows are zero log-odds (neutral: contributes nothing to the evo
    term), used when the alignment spans fewer residues than the design unit
    (e.g. RNAalign covered only the first chain of a multi-chain unit).
    """
    L = pssm.shape[0]
    if L == target_length:
        return pssm
    if L > target_length:
        return pssm[:target_length]
    pad = np.zeros((target_length - L, pssm.shape[1]), dtype=pssm.dtype)
    return np.vstack([pssm, pad])


@dataclass
class GenerateProfileConfig:
    input_pdb: str
    output_dir: str = "./data/test_output"
    bin_path: str = str(Path(__file__).resolve().parent)
    pdb_lib: str = ""
    cutoff: float = 0.45   # include all homologues with TM-score > cutoff
    write_msa_file: bool = True
    max_template_len: int = 1500   # skip templates larger than this (nt); 0 disables
    min_template_len: int = 6      # skip templates with fewer nt (RNAalign needs >5); 0 disables


class RNAligningPipeline:
    """
    Align library PDBs to the input structure; produces ``RnaAlignmentHit`` list
    and optional tmscore / msa artifacts.
    """

    def __init__(self, config: GenerateProfileConfig) -> None:
        self.cfg = config
        self.input_pdb = Path(config.input_pdb).resolve()
        self.output_dir = Path(config.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._rnaalign = Path(config.bin_path) / "RNAalign" / "RNAalign"
        self.pdb_lib = Path(config.pdb_lib).resolve() if config.pdb_lib else self.output_dir
        self.write_msa_file = config.write_msa_file
        self.alignments: List[RnaAlignmentHit] = []

    def run_rnaalign(self) -> List[RnaAlignmentHit]:
        decoys = sorted(self.pdb_lib.glob("*.pdb")) + sorted(self.pdb_lib.glob("*.cif"))
        logger.info("Structure library: %s  (%d templates found)", self.pdb_lib, len(decoys))
        logger.info("RNAalign TM-score cutoff: %.2f", self.cfg.cutoff)

        tmscore_lines: List[str] = []
        seqid_lines: List[str] = []
        filtered_results: List[RnaAlignmentHit] = []
        n_searched = 0
        n_skipped_large = 0
        n_skipped_short = 0

        for decoy in decoys:
            pdb_id = decoy.stem
            # Peek the template size first: RNAalign cannot parse very large chains
            # (rRNA / atom-serial overflow → "Chain number 0") and refuses chains of
            # ≤5 residues ("sequence too short").  Skip both up front instead of
            # invoking RNAalign and parsing its warnings.  n_res == -1 (unreadable)
            # passes through so RNAalign still gets a chance.
            n_res = _quick_residue_count(decoy)
            if self.cfg.max_template_len and n_res > self.cfg.max_template_len:
                n_skipped_large += 1
                continue
            if self.cfg.min_template_len and 0 <= n_res < self.cfg.min_template_len:
                n_skipped_short += 1
                continue
            n_searched += 1
            result = self._align_one(pdb_id, decoy)
            if result is None:
                continue
            tmscore_lines.append(f"{result.tm_score:.6f}\t{pdb_id}")
            seqid_lines.append(f"{result.seq_id:.6f}\t{pdb_id}")

            if result.tm_score <= self.cfg.cutoff:
                continue

            filtered_results.append(result)

        self._write_seqid_file(seqid_lines, self.output_dir / "seqid.txt")
        self._write_tmscore_file(tmscore_lines, self.output_dir / "tmscore.txt")
        self._write_tmscore_file(tmscore_lines, self.output_dir / "tmscore_sorted.txt", do_sort=True)

        if self.cfg.write_msa_file:
            write_msa_txt(self.output_dir / "msa.txt", filtered_results)

        if n_skipped_large or n_skipped_short:
            logger.info("Skipped %d oversized (> %d nt) and %d too-short (< %d nt) templates before RNAalign",
                        n_skipped_large, self.cfg.max_template_len,
                        n_skipped_short, self.cfg.min_template_len)
        logger.info("Templates searched: %d, passed cutoff: %d", n_searched, len(filtered_results))
        self.alignments = fetch_top_hits(filtered_results)  # removes perfect self-matches, sorts by TM-score

    def _align_one(self, pdb_id: str, decoy: Path) -> Optional[RnaAlignmentHit]:
        cmd = [str(self._rnaalign), str(decoy), str(self.input_pdb)]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            logger.warning("RNAalign timed out for %s — skipping", pdb_id)
            return None
        except FileNotFoundError:
            logger.error("RNAalign binary not found at: %s", self._rnaalign)
            sys.exit(1)

        # Some builds write only to stdout, others split across stdout/stderr; merge for robustness.
        combined = (proc.stdout or "") + (proc.stderr or "")
        if "sequence is too short" in combined.lower():
            logger.warning(
                "RNAalign did not run alignment for template %s (sequence too short, ≤5 residues); skipping",
                pdb_id,
            )
            return None

        lines = combined.splitlines()
        return self._parse_rnalign_output(pdb_id, self.input_pdb.stem, lines)

    @staticmethod
    def _parse_rmsd_and_aln(lines: List[str], projected_query_len: int) -> Tuple[float, int]:
        text = "\n".join(lines)
        rmsd = 0.0
        m = re.search(r"RMSD\s*=\s*([0-9.]+)", text, re.I)
        if m:
            try:
                rmsd = float(m.group(1))
            except ValueError:
                pass
        aln = 0
        m = re.search(r"\baln\s*=\s*(\d+)", text, re.I)
        if m:
            try:
                aln = int(m.group(1))
            except ValueError:
                pass
        if aln <= 0:
            aln = projected_query_len
        return rmsd, aln

    @staticmethod
    def _parse_rnalign_output(
        pdb_id: str, query_id: str, lines: List[str]
    ) -> Optional[RnaAlignmentHit]:
        """Parse RNAalign default-format stdout (TM-score, Seq_ID line, gapped alignment block)."""
        text = "\n".join(lines)

        tm_m = re.search(r"TM-score=\s*([0-9.]+)", text)
        if not tm_m:
            logger.warning(
                "No TM-score= in RNAalign output for %s (parse error or run failed); skipping: %s",
                pdb_id, 
                text,
            )
            return None
        try:
            tm_score = float(tm_m.group(1))
        except ValueError:
            logger.warning("Could not parse TM-score float for %s", pdb_id)
            return None

        seq_id = 0.0
        seq_m = re.search(r"Seq_ID=n_identical/n_aligned=\s*([0-9.]+)", text)
        if seq_m:
            try:
                seq_id = float(seq_m.group(1))
            except ValueError:
                pass
        else:
            hdr_m = re.search(r"(?:^|\t)seqID=([0-9.]+)", text, re.MULTILINE | re.IGNORECASE)
            if hdr_m:
                try:
                    seq_id = float(hdr_m.group(1))
                except ValueError:
                    pass

        template_ali: Optional[str] = None
        query_ali: Optional[str] = None
        for j, line in enumerate(lines):
            if "denotes other aligned residues" not in line:
                continue
            try:
                template_ali = lines[j + 1].strip()
                query_ali = lines[j + 3].strip()
            except IndexError:
                logger.warning("Alignment block truncated for %s", pdb_id)
                return None
            break

        if template_ali is None or query_ali is None:
            return None

        query_ali_full = query_ali.strip()
        template_ali_full = template_ali.strip()

        projected: List[str] = []
        projected_tmpl: List[str] = []
        for t_char, q_char in zip(template_ali, query_ali):
            if t_char != "-":
                projected.append(q_char)
                projected_tmpl.append(t_char)

        projected_q = "".join(projected)
        rmsd, aln = RNAligningPipeline._parse_rmsd_and_aln(lines, len(projected_q))

        return RnaAlignmentHit(
            query_id=query_id,
            target_id=pdb_id,
            tm_score=tm_score,
            seq_id=seq_id or 0.0,
            rmsd=rmsd,
            aln=aln,
            query_row=query_ali_full,
            target_row=template_ali_full,
            query_aligned=projected_q,
            template_aligned="".join(projected_tmpl),
        )

    @staticmethod
    def _write_tmscore_file(
        lines: List[str], path: Path, do_sort: bool = False
    ) -> None:
        if do_sort:
            lines = sorted(lines, key=lambda x: float(x.split("\t")[0]), reverse=True)
        path.write_text("\n".join(lines) + "\n")

    @staticmethod
    def _write_seqid_file(lines: List[str], path: Path) -> None:
        path.write_text("\n".join(lines) + "\n")

    @staticmethod
    def _write_prf_file(w, path: Path) -> None:
        with open(path, "w") as f:
            f.write("pos a u g c\n")
            for i in range(w.shape[0]):
                f.write(f"{i+1} ")
                for j in range(w.shape[1]):
                    f.write(f"{w[i, j]:.6f} ")
                f.write("\n")

    def get_pssm(self, target_length: int | None = None) -> np.ndarray:
        """Log-odds PSSM (L, 4).

        RNAalign aligns the query as the structure presents it; for a
        multi-chain design unit it only spans the first chain, so the alignment
        length can be shorter than the full unit (e.g. chain A = 13 of a 21-nt
        A-B unit).  Pass *target_length* (the design-unit length) to pad the
        uncovered tail with neutral (zero log-odds) rows so the PSSM matches the
        ``Structure`` the MC loop scores — otherwise the evo term silently
        misaligns / the validator warns ``pssm shape (13,4) != (21,4)``.
        """
        L = len(self.alignments[0].query_row)
        pssm = compute_M(self.alignments, L)
        if target_length is not None and target_length != L:
            pssm = _fit_length(pssm, target_length)
        write_prf(pssm, self.output_dir / "prf.txt")
        return pssm


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PSSM weights from RNAalign hits")
    parser.add_argument("input_pdb", type=str, help="Input PDB or mmCIF file")
    parser.add_argument("--output_dir", type=str, default="./data/test_output")
    parser.add_argument("--bin_path", type=str, default="./bin")
    parser.add_argument("--pdb_lib", type=str, default="", help="Directory of template structures")
    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.45,
        help="TM-score cutoff — all homologues above this are used (default: %(default)s)",
    )
    parser.add_argument(
        "--no-msa-file",
        action="store_true",
        help="Do not write msa.txt (only in-memory hits → compute_w)",
    )
    args = parser.parse_args()

    config = GenerateProfileConfig(
        input_pdb=args.input_pdb,
        output_dir=args.output_dir,
        bin_path=args.bin_path,
        pdb_lib=args.pdb_lib,
        cutoff=args.cutoff,
        write_msa_file=not args.no_msa_file,
    )

    pipeline = RNAligningPipeline(config)
    pipeline.run_rnaalign()
    pssm = pipeline.get_pssm()
    logger.info("PSSM shape: %s  (%d homologues used)", pssm.shape, len(pipeline.alignments))
    structure = build_structure(
        args.input_pdb,
        args.output_dir,
        alignments=pipeline.alignments,
        n_alignments=len(pipeline.alignments),
    )
    logger.info("Structure built: %s", structure)



if __name__ == "__main__":
    main()
