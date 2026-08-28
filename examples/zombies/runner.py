# %%
# Main script to run the simulation
import time
import json
import os
from .Builder import Builder

#%%

#import tracemalloc
#tracemalloc.start()



def run_simulation(model_time, person_quantity, person_speed, zombie_quantity, zombie_speed, progress_bar):
    #print("Starting simulation...")
    b = Builder(model_time, person_quantity, person_speed, zombie_quantity, zombie_speed, progress_bar)

    #print("Builder created. Running simulation...")

    (kpi, repo) = b.engine.run(2)
    #print("Simulation run completed. Exporting data...")

    print("KPIs collected:")
    for k, v in kpi.items():
        print(f"{k}: {v}")

    toc = time.perf_counter()
    print(f"Simulation completed in {toc - tic:0.4f} seconds")

    # save repo to json file with warning if file already exists
    file_path = "zombie_simulation_data.json"
    if os.path.exists(file_path):
        print(f"Warning: {file_path} already exists and will be overwritten.")

    # Write repo to JSON file with indentation
    with open(file_path, "w") as f:
        json.dump(repo, f, indent=4)


# %%

#snapshot = tracemalloc.take_snapshot()
#top = snapshot.statistics('lineno')
#for stat in top[:10]:
#    print(stat)

# Visualize the simulation data

# Animate from the saved JSON file
# anim = vc.animate_repo_from_json("zombie_simulation_data.json", interval=10, carry_forward=True, figsize=(12,9))


if __name__ == "__main__":
    tic = time.perf_counter()

    model_time=1000
    person_quantity=10
    person_speed=1
    zombie_quantity=1
    zombie_speed=0.01
    progress_bar = True

    run_simulation(model_time, person_quantity, person_speed, zombie_quantity, zombie_speed, progress_bar)