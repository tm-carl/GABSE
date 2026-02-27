# %%
# Main script to run the simulation
import time
import json
import os
import sys

# Ensure the repo src directory is on sys.path so worker processes can import local packages
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
# Also set PYTHONPATH environment variable so spawned processes inherit it
os.environ.setdefault('PYTHONPATH', SRC_PATH)



#%%

from concurrent.futures import ProcessPoolExecutor
import itertools


levels = {
    "model_time": range(1000, 10000, 10000),
    "person_quantity": range(10, 100, 100),
    "person_speed": range(1, 10, 1),
    "zombie_quantity": range(10, 100, 10),
    "zombie_speed": range(1, 10, 1)
}


keys = list(levels.keys())
design = [dict(zip(keys, vals)) for vals in itertools.product(*levels.values())]


def simulate(scenario):
    import Builder
    model_time = scenario["model_time"]
    person_quantity = scenario["person_quantity"]
    person_speed = scenario["person_speed"]
    zombie_quantity = scenario["zombie_quantity"]
    zombie_speed = scenario["zombie_speed"]

    sim = Builder.Builder(model_time, person_quantity, person_speed, zombie_quantity, zombie_speed)
    kpi = sim.engine.run(1)
    return scenario, kpi


def run_design(design_list):
    results = []
    with ProcessPoolExecutor() as executor:
        for res in executor.map(simulate, design_list):
            results.append(res)
    return results


if __name__ == "__main__":
    # Required on Windows to safely start child processes
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except Exception:
        pass

    tic = time.perf_counter()
    print("Starting simulations...")

    results = run_design(design)
    print("Number of results:", len(results))

    toc = time.perf_counter()
    print(f"Simulation completed in {toc - tic:0.4f} seconds")

    # Quick single-run smoke test (useful when debugging import/Builder issues)
    #test_scenario = {"model_time": 100.0, "person_quantity": 5, "person_speed": 1, "zombie_quantity": 1, "zombie_speed": 1}
    #scenario, kpis = simulate(test_scenario)
    #print("Test run KPIs:", kpis)
