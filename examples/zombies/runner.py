# %%
# Main script to run the simulation
import time

import Builder

tic = time.perf_counter()

b = Builder.Builder()
b.engine.run()
b.dataLogger.collect_data()
repo = b.dataLogger.export_data()

toc = time.perf_counter()
print(f"Simulation completed in {toc - tic:0.4f} seconds")

# save repo to json file
import json

with open("zombie_simulation_data.json", "w") as f:
    json.dump(repo, f, indent=4)


# %%

# Visualize the simulation data

# Animate from the saved JSON file
#anim = vc.animate_repo_from_json("zombie_simulation_data.json", interval=10, carry_forward=True, figsize=(12,9))
