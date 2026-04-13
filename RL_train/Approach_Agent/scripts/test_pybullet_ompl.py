import os
import sys

from src.pybullet_ompl import main


if __name__ == "__main__":
    # OMPL bindings can crash during Python interpreter shutdown in some
    # environments after planning has already completed successfully.
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
