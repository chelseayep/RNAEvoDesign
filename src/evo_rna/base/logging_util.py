"""
Centralized logging for evo_rna pipeline and cluster runs.

Configure once at job start (e.g. in read_pdb.py or your driver), then use
get_logger(__name__) in any module to log. Logs go to stderr by default;
when a log file is set, the same messages are also written to that file
with timestamps and levels for later debugging.

Cluster / batch usage:
  - Pass --log-file pipeline.log so each run writes to run_dir/pipeline.log.
  - Set EVO_RNA_LOG_LEVEL=DEBUG in the job script for verbose subprocess and
    path logging when diagnosing failures.
  - Set EVO_RNA_JOB_ID=$SLURM_JOB_ID (or similar) to tag each line with the
    job id in the log file.

Environment variables (optional):
  EVO_RNA_LOG_FILE   path to log file (enables file logging)
  EVO_RNA_LOG_LEVEL  DEBUG | INFO | WARNING | ERROR (default: INFO)
  EVO_RNA_JOB_ID     optional job/slurm id to include in log lines
"""

import logging
import os
import sys
from typing import Optional

# Logger name used by pipeline scripts and preprocess
LOGGER_NAME = "evo_rna"

# Format with timestamp, level, optional job id, and message
# Cluster-friendly: one line per record, easy to grep
DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)
FORMAT_WITH_JOB = (
    "%(asctime)s | %(levelname)-7s | job=%(job_id)s | %(name)s | %(message)s"
)


class JobIdFilter(logging.Filter):
    """Add job_id to the LogRecord if set (for cluster logs)."""
    def __init__(self, job_id: Optional[str] = None):
        super(JobIdFilter, self).__init__()
        self.job_id = job_id or os.environ.get("EVO_RNA_JOB_ID", "")

    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = self.job_id or "-"
        return True


def configure_logging(
    log_file: Optional[str] = None,
    log_level: Optional[str] = None,
    job_id: Optional[str] = None,
    logger_name: str = LOGGER_NAME,
    use_env: bool = True,
) -> logging.Logger:
    """
    Configure logging for the pipeline. Call once at the start of a run.

    Parameters
    ----------
    log_file : str or None
        If set, append all log records to this file (e.g. run_dir/pipeline.log).
        Also read from EVO_RNA_LOG_FILE if use_env and not passed.
    log_level : str or None
        One of DEBUG, INFO, WARNING, ERROR. Default INFO.
        Also read from EVO_RNA_LOG_LEVEL if use_env and not passed.
    job_id : str or None
        Optional cluster job id to include in each line. Also read from
        EVO_RNA_JOB_ID if use_env and not passed.
    logger_name : str
        Name of the logger to configure (default: "evo_rna").
    use_env : bool
        If True, fill log_file / log_level / job_id from env when not passed.

    Returns
    -------
    logging.Logger
        The configured logger. Use this or get_logger(logger_name) elsewhere.
    """
    if use_env:
        log_file = log_file or os.environ.get("EVO_RNA_LOG_FILE")
        log_level = log_level or os.environ.get("EVO_RNA_LOG_LEVEL", "INFO").upper()
        job_id = job_id or os.environ.get("EVO_RNA_JOB_ID")
    else:
        log_level = log_level or "INFO"

    logger = logging.getLogger(logger_name)
    # Avoid re-adding handlers if already configured (e.g. multiple runs in same process)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Formatter: include job_id in format if set
    jid = job_id or os.environ.get("EVO_RNA_JOB_ID")
    if jid:
        formatter = logging.Formatter(FORMAT_WITH_JOB)
        job_filter = JobIdFilter(jid)
    else:
        formatter = logging.Formatter(DEFAULT_FORMAT)
        job_filter = JobIdFilter("")

    logger.addFilter(job_filter)

    # Console handler (stderr)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Optional file handler (same format, append)
    if log_file:
        try:
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.info("Log file: %s", os.path.abspath(log_file))
        except OSError as e:
            logger.warning("Could not open log file %s: %s", log_file, e)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger that propagates to the configured evo_rna logger.

    Use in modules that don't call configure_logging themselves, e.g.:
        from evo_rna.base.logging_util import get_logger
        log = get_logger(__name__)
        log.info("message")
    """
    if name.startswith(LOGGER_NAME):
        return logging.getLogger(name)
    # Child loggers (e.g. evo_rna.io.program_preprocess) propagate to evo_rna
    return logging.getLogger(name)
