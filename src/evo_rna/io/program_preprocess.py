# Converted from src/io/ProgramPreprocess.h and ProgramPreprocess.cpp

import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from evo_rna.base.logging_util import get_logger

log = get_logger(__name__)

# Module-level config (set by caller or pipeline)
PROGRAM_PATH: str = ""
PROGRAM_NAME: str = ""
PROGRAM_VERSION: str = ""
PDBID: str = ""
TGT_PRF: str = "target.prf"
TGT_MSA: str = "target.msa"
DES_CHAINS: str = "A"
PDBLIB: str = ""  # PDB template directory for TM-align (set by read_pdb.py --pdblib)
TMCUTOFF: Optional[float] = None  # TM-score cutoff (set by read_pdb.py --tmcutoff)
PROT_PROFILE: Optional[list] = None  # list of list of float (per-residue profile)

# Constants (from pipeline Utility.h)
MAX_LEN_ONE_LINE_CONTENT = 1024
MAX_LEN_FILE_NAME = 100
MAX_LEN_ERR_MSG = 1024


def _bin_path() -> str:
    """
    Resolve the bin/ directory.  Priority:
      1. PROGRAM_PATH if it already points at (or contains) a bin/ subdir
      2. The bin/ directory sitting next to the package root (project root)
    """
    if PROGRAM_PATH:
        candidate = os.path.join(PROGRAM_PATH, "bin")
        if os.path.isdir(candidate):
            return candidate
        if os.path.isdir(PROGRAM_PATH):
            return PROGRAM_PATH
    # Walk up from this file to find the project-level bin/
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "bin"
        if candidate.is_dir():
            return str(candidate)
    return "bin"


def extract_path_and_name(fullpath: str) -> Tuple[str, str]:
    """
    Split full path into directory path and base name.
    Returns (path, name). C: ExtractPathAndName(fullpath, path, name).
    """
    path = os.path.dirname(fullpath)
    name = os.path.basename(fullpath)
    if not path or path == ".":
        path = "."
    return (path, name)


def get_pdb_id(pdbfile: str) -> Tuple[int, str]:
    """
    Extract structure ID from filename (strip .pdb or .cif). Returns (Success/Warning, pdbid).
    If file doesn't end in .pdb or .cif, returns Warning and "yourpdb".
    """
    if not pdbfile:
        return (1, "yourpdb")  # Warning
    p = Path(pdbfile)
    if p.suffix.lower() not in (".pdb", ".cif"):
        return (1, "yourpdb")
    return (0, p.stem)  # Success


def generate_profile(pdb: str, pdblib: str = "", tm_cutoff: float = 0.8) -> int:
    """
    Run GenerateProfile.pl to build an evolutionary profile for *pdb*.

    The script orchestrates:
      TMalign  – structure alignment against a PDB library
      PDB2FAS  – extract FASTA from a PDB file
      psiblast – PSI-BLAST homology search (ncbi-blast-2.7.1+/bin/psiblast)
      xml2msa.pl – convert BLAST XML output to an MSA text file
      mkprf    – build a position-specific scoring matrix from the MSA

    Outputs written to the current working directory:
      msa_tmalign.txt, msa_tmalign.fas, msa_psiblast.txt, msa.txt,
      tmscore.txt, seqid.txt, prf.txt

    Uses module-level PDBLIB and TMCUTOFF when not passed (e.g. when called
    from structure_deploy_evolution_info after read_pdb.py set them via --pdblib).
    """
    bin_dir = _bin_path()
    script = os.path.join(bin_dir, "GenerateProfile.pl")
    if not os.path.isfile(script):
        log.warning("GenerateProfile.pl not found at %s", script)
        return -1  # binary not found

    effective_pdblib = pdblib or PDBLIB
    effective_tm = tm_cutoff if TMCUTOFF is None else TMCUTOFF

    cmd = ["perl", script]
    if effective_pdblib:
        cmd += ["--pdblib", effective_pdblib]
    if effective_tm != 0.8:
        cmd += ["--tmcutoff", str(effective_tm)]
    cmd.append(pdb)

    cwd = os.path.abspath(".")
    log.info("generate_profile: cwd=%s pdblib=%s tm_cutoff=%s", cwd, effective_pdblib or "(default)", effective_tm)
    # Warn when using default library that doesn't exist or is empty (so TMalign is never called)
    if not effective_pdblib:
        _default_pdblib = "/nfs/amino-library/PDB"
        if not os.path.isdir(_default_pdblib):
            log.warning(
                "Default PDB library %s does not exist; no TM-align templates. "
                "TMalign will not be run. Use --pdblib with a directory that has .pdb files.",
                _default_pdblib,
            )
        else:
            n_pdb = len([f for f in os.listdir(_default_pdblib) if f.lower().endswith((".pdb", ".cif"))])
            if n_pdb == 0:
                log.warning(
                    "Default PDB library %s has 0 structure files (.pdb/.cif); TMalign will not be run. Use --pdblib.",
                    _default_pdblib,
                )
    log.debug("generate_profile cmd: %s", " ".join(cmd))
    if os.environ.get("EVO_RNA_VERBOSE_PROFILE"):
        log.info("GenerateProfile.pl cwd=%s cmd=%s", cwd, " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            cwd=".",
            timeout=3600,
        )
        log.info("generate_profile finished with returncode=%s", result.returncode)
        return result.returncode
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        log.warning("generate_profile subprocess failed: %s", e)
        return -1


def read_profile(seq_length: int) -> int:
    """
    Read profile from TGT_PRF into PROT_PROFILE (global).
    *seq_length*: number of residue rows expected.
    """
    global PROT_PROFILE
    try:
        with open(TGT_PRF) as f:
            lines = f.readlines()
        PROT_PROFILE = []
        for line in lines:
            toks = line.split()
            if len(toks) <= 1:
                continue
            try:
                float(toks[1])
            except (ValueError, IndexError):
                continue
            row = [float(t) for t in toks[1:]]
            PROT_PROFILE.append(row)
        log.info("read_profile: loaded %s rows from %s", len(PROT_PROFILE), TGT_PRF)
    except OSError as e:
        PROT_PROFILE = []
        log.debug("read_profile: could not read %s: %s", TGT_PRF, e)
    return 0


def generate_single_sequence_features(pdb: str) -> int:
    """
    Run the DSSP pipeline to generate secondary-structure, solvent-accessibility,
    and phi/psi dihedral files for *pdb*.

    Steps
    -----
    1. dsspcmbi <pdb>          → dssp.txt   (raw DSSP output)
    2. dssp-ver2hor dssp.txt   → ss.txt     (horizontal SS string: H/E/C)
    3. convSS.pl ss.txt        → sa.txt, phi-psi.txt, ss.txt (I-TASSER format)

    All output files are written to the current working directory.
    """
    bin_dir = _bin_path()

    dsspcmbi = os.path.join(bin_dir, "dsspcmbi")
    if not os.path.isfile(dsspcmbi):
        log.warning("dsspcmbi not found at %s", dsspcmbi)
        return -1

    log.debug("generate_single_sequence_features: running dsspcmbi on %s", pdb)
    try:
        with open("dssp.txt", "w") as dssp_out:
            result = subprocess.run(
                [dsspcmbi, pdb],
                stdout=dssp_out,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
        if result.returncode != 0:
            log.info("dsspcmbi exit code %s", result.returncode)
            return result.returncode
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        log.warning("dsspcmbi failed: %s", e)
        return -1

    # Step 2: dssp-ver2hor – convert vertical DSSP output to a single     #
    #         horizontal SS string written to ss.txt                      #

    dssp_ver2hor = os.path.join(bin_dir, "dssp-ver2hor")
    if os.path.isfile(dssp_ver2hor):
        try:
            with open("ss.txt", "w") as ss_out:
                result = subprocess.run(
                    [dssp_ver2hor, "dssp.txt"],
                    stdout=ss_out,
                    stderr=subprocess.DEVNULL,
                    timeout=60,
                )
            if result.returncode != 0:
                log.info("dssp-ver2hor exit code %s", result.returncode)
                return result.returncode
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            log.warning("dssp-ver2hor failed: %s", e)
            return -1
    else:
        log.debug("dssp-ver2hor not found, skipping ss.txt")


    # Step 3: convSS.pl – convert SS to I-TASSER numeric encoding and     #
    #         extract sa.txt / phi-psi.txt from dssp.txt                  #

    conv_ss = os.path.join(bin_dir, "convSS.pl")
    if os.path.isfile(conv_ss) and os.path.isfile("ss.txt"):
        try:
            with open("ss_itasser.txt", "w") as ss_itasser_out:
                result = subprocess.run(
                    ["perl", conv_ss, "ss.txt"],
                    stdout=ss_itasser_out,
                    stderr=subprocess.DEVNULL,
                    timeout=60,
                )
            if result.returncode != 0:
                log.info("convSS.pl exit code %s", result.returncode)
                return result.returncode
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            log.warning("convSS.pl failed: %s", e)
            return -1
    else:
        log.debug("convSS.pl or ss.txt missing, skipping ss_itasser/sa/phi-psi")

    return 0


def _write_structure_pdb(structure, filepath: str) -> None:
    """Write a core.Structure's atoms to a minimal PDB file."""
    with open(filepath, "w") as fh:
        for serial, atom in enumerate(structure.atoms, start=1):
            # Minimal ATOM record — enough for DSSP / TMalign
            fh.write(
                f"ATOM  {serial:5d}  {atom.name:<3s}   {structure.chain_id:1s}"
                f"        {atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}\n"
            )


def structure_deploy_evolution_info(structure) -> int:
    """
    Deploy evolution-related data: write target chain PDB, generate/read profile,
    generate single-sequence features.

    Accepts the new flat core.Structure.
    """
    global PROT_PROFILE, PDBID, TGT_PRF, TGT_MSA, DES_CHAINS

    if structure.length == 0:
        log.warning("structure_deploy_evolution_info: empty structure")
        return -2

    tgt_chain_file = f"{PDBID}_TGTCHN.pdb"
    log.info(
        "structure_deploy_evolution_info: writing %s (%d residues)",
        tgt_chain_file,
        structure.length,
    )
    try:
        _write_structure_pdb(structure, tgt_chain_file)
    except OSError as e:
        log.warning("Failed to write %s: %s", tgt_chain_file, e)
        return -2

    res_count = structure.length

    # Check if profile exists and has correct length; regenerate if needed
    line_count = 0
    try:
        with open(TGT_PRF) as f:
            line_count = sum(1 for _ in f)
    except OSError:
        pass

    if line_count < res_count or line_count > res_count + 2:
        log.info(
            "Profile/MSA missing or wrong length (lines=%s residues=%s), regenerating",
            line_count,
            res_count,
        )
        try:
            with open(TGT_MSA) as msa:
                msa_lines = sum(1 for _ in msa)
            if msa_lines != res_count:
                generate_profile(tgt_chain_file)
        except OSError:
            generate_profile(tgt_chain_file)

    log.debug("Running generate_single_sequence_features on %s", tgt_chain_file)
    generate_single_sequence_features(tgt_chain_file)
    return 0
