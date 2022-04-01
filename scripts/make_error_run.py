
import json
import numpy as np
import itertools
import os

if __name__ == "__main__":

    N_cells = (8,16,32)
    p = (1,2,4,8)
    N = 16
    #N_particles = np.logspace(3, 11, N, dtype="int")
    #N_particles = (3,4,5,6,7,8,9,10,11)
    N_particles = (7,8,9,10,11)

    run_index = 0

    for runx in itertools.product(
        N_cells, p, N_particles
        ):
        run_index += 1
        
        path = os.path.join(os.getcwd(), f"run_{run_index}")
        os.mkdir(path)

        config = {
            "N_cells": int(runx[0]),
            "p": int(runx[1]),
            "N_particles": int(runx[2]),
        }

        open(os.path.join(path, "input.json"), "w").write(json.dumps(config))

