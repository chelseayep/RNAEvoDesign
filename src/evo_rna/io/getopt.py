# Python conversion of src/io/Getopt.h / Getopt.cpp (GNU getopt_long-style API)
# For full command-line parsing, consider using argparse.

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# Names for the values of the `has_arg' field of `struct option'.
no_argument = 0
required_argument = 1
optional_argument = 2

# Module-level state (like C's extern optarg, optind, opterr, optopt)
optarg: Optional[str] = None
optind: int = 1
opterr: int = 1
optopt: int = ord("?")

# Internal reset flag (optreset in some C implementations)
optreset: int = 0

BADCH = ord("?")
BADARG = ord(":")  # or '?' depending on options string
INORDER = 1


@dataclass
class Option:
    """Long option descriptor (struct option)."""
    name: str
    has_arg: int  # no_argument, required_argument, optional_argument
    flag: Optional[List[int]] = None  # if set, *flag = val when option found
    val: int = 0  # value to return or store in *flag

    def __post_init__(self) -> None:
        if self.flag is None:
            self.flag = None  # C NULL


def _find_long(
    argv: Sequence[str],
    arg: str,
    longopts: List[Option],
    longind: Optional[List[int]],
) -> Tuple[int, Optional[str], Optional[str]]:
    """Parse long option. Returns (val_or_0, optarg, error_msg)."""
    global optarg, optopt
    if not arg.startswith("--"):
        return (-1, None, None)
    current = arg[2:]
    has_equal = current.find("=")
    if has_equal >= 0:
        optarg = current[has_equal + 1 :]
        current = current[:has_equal]
    else:
        optarg = None

    match_idx = -1
    for i, opt in enumerate(longopts):
        if not opt.name:
            break
        if not current.startswith(opt.name):
            continue
        if len(opt.name) == len(current):
            match_idx = i
            break
        if match_idx >= 0:
            if opterr:
                return (BADCH, None, f"ambiguous option -- {current}\n")
            match_idx = -1
            break
        match_idx = i

    if match_idx < 0:
        if opterr:
            return (BADCH, None, f"unknown option -- {arg}\n")
        return (-1, None, None)

    opt = longopts[match_idx]
    if opt.has_arg == no_argument and has_equal >= 0:
        if opterr:
            return (BADARG, None, f"option doesn't take an argument -- {arg}\n")
        optopt = opt.val if opt.flag is None else 0
        return (BADARG, None, None)
    if opt.has_arg == required_argument and optarg is None:
        optarg = None  # will be set by caller from next argv
    if longind is not None:
        longind.append(match_idx)
    if opt.flag:
        opt.flag[0] = opt.val
        return (0, optarg, None)
    return (opt.val, optarg, None)


def getopt(
    argc: int,
    argv: Sequence[str],
    shortopts: str,
) -> int:
    """
    Parse next short option from argv. Sets optarg, optind, optopt.
    Returns option character, or -1 when done, or BADCH/BADARG on error.
    """
    global optarg, optind, optopt, optreset
    optarg = None
    if optind >= argc or optind < 0:
        return -1
    arg = argv[optind]
    if not arg or arg == "--":
        return -1
    if arg[0] != "-" or (len(arg) == 1 and "-" not in shortopts):
        return -1
    if arg == "-":
        optind += 1
        return getopt(argc, argv, shortopts)
    if arg.startswith("--"):
        optind += 1
        return -1  # long options handled by getopt_long

    # short option(s)
    ch = arg[1]
    optopt = ord(ch)
    col = shortopts.find(ch)
    if col < 0:
        if opterr:
            import sys
            print(f"unknown option -- {ch}", file=sys.stderr)
        return BADCH
    if col + 1 < len(shortopts) and shortopts[col + 1] == ":":
        if len(arg) > 2:
            optarg = arg[2:]
        else:
            optind += 1
            if optind >= argc:
                if opterr:
                    import sys
                    print(f"option requires an argument -- {ch}", file=sys.stderr)
                return BADARG
            optarg = argv[optind]
        optind += 1
        return optopt
    if len(arg) > 2:
        # push back remainder into argv by advancing only past this char
        # C code uses static 'place'; we mutate a copy of argv or track position
        optind += 1
        return optopt
    optind += 1
    return optopt


def getopt_long(
    argc: int,
    argv: Sequence[str],
    shortopts: str,
    longopts: Sequence[Option],
    longind: Optional[List[int]] = None,
) -> int:
    """
    Parse next option (short or long). Sets optarg, optind, optopt.
    Returns option val (or short char), 0 if flag was set, -1 when done, BADCH/BADARG on error.
    """
    global optarg, optind, optopt
    optarg = None
    if longind is not None:
        longind.clear()
    if optind >= argc or optind < 0:
        return -1
    arg = argv[optind]
    if not arg or arg == "-":
        return -1
    if arg == "--":
        optind += 1
        return -1
    if arg.startswith("--"):
        optind += 1
        longopts_list = list(longopts)
        val, next_arg, err = _find_long(argv, arg, longopts_list, longind)
        if err and opterr:
            import sys
            sys.stderr.write(err)
        if val == 0:
            return 0
        if val == BADCH or val == BADARG:
            return val
        if next_arg is not None:
            optarg = next_arg
        else:
            idx = longind[0] if longind and len(longind) > 0 else -1
            if idx >= 0 and idx < len(longopts_list):
                o = longopts_list[idx]
                if o.has_arg == required_argument and optind < argc:
                    optarg = argv[optind]
                    optind += 1
        return val
    if arg[0] == "-":
        return getopt(argc, argv, shortopts)
    return -1


def getopt_long_only(
    argc: int,
    argv: Sequence[str],
    shortopts: str,
    longopts: Sequence[Option],
    longind: Optional[List[int]] = None,
) -> int:
    """Like getopt_long but single '-' also triggers long option parsing."""
    return getopt_long(argc, argv, shortopts, longopts, longind)
