import os
import sys
from pathlib import Path

from src.utils.utils import read_json, write_json

curdir = os.path.dirname(__file__)
curdir = Path(curdir)


pclprof = read_json("/home/bulat/PycharmProjects/edge-based-diffusion/main.py.pclprof")

if __name__ == "__main__":

    if pclprof.get('fixed'):
        sys.exit("Already processed")

    for j, functions in enumerate(pclprof["profiledFunctions"]):

        p = Path(functions["file"])

        i = [i + 1 for i, p_ in enumerate(p.parts) if p_ == "edge-based-diffusion"][0]
        relative_dir = Path(*list(p.parts[i:]))

        functions["file"] = str(Path.joinpath(curdir, relative_dir))
        functions["unit"] = 1e-03

        for k, line in enumerate(functions["profiledLines"]):

            line["time"] = int( line["time"] / 1e6)

            # pack everything
            functions['profiledLines'][k] = line
        pclprof["profiledFunctions"][j] = functions

    pclprof['unit'] = pclprof['unit'] = 1e-03
    pclprof['fixed'] = True

    write_json(pclprof, "/home/bulat/PycharmProjects/edge-based-diffusion/main.py.pclprof")