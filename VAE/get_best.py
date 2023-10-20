import json
import os
import numpy as np

# list of all subdirectories in tmp_eval folder
subdirs = [x[0] for x in os.walk('tmp_eval/')][1:]
loss_list = []

for subdir in subdirs:
    with open(os.path.join(subdir, 'config.json')) as f:
        metrics = json.load(f)
        loss = metrics['lowest_loss']
        loss_list.append((subdir, loss))

sorted_loss = sorted(loss_list, key=lambda x: x[1])
print(f"Best model: {sorted_loss[0][0]} with loss {round(sorted_loss[0][1],2)}\n")
print("Complete ranking:")
for i in range(len(sorted_loss)):
    print(f"{i+1}. {sorted_loss[i][0]} with loss {round(sorted_loss[i][1],2)}")