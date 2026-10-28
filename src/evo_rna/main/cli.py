"""Console-script entry point for ``rna-evo-design``.

``design.py`` is a straight-line script: it parses argv and runs the pipeline at
module level, which is what makes it readable top to bottom. That shape has no
``main()`` to point an entry point at, so this launcher executes it under
``__main__`` instead of restructuring the pipeline around a function boundary.
"""

from __future__ import annotations

import runpy


def main() -> None:
    runpy.run_module("evo_rna.main.design", run_name="__main__")


if __name__ == "__main__":
    main()
