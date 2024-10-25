import json
import numpy as np

# Load the JSON data
file_path = 'NuScenes_FasterRCNN/results.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract metrics for VEHICLE, PEDESTRIAN, and CYCLIST for precision, recall, and ap50
vehicle_precision = []
pedestrian_precision = []
cyclist_precision = []

vehicle_recall = []
pedestrian_recall = []
cyclist_recall = []

vehicle_ap50 = []
pedestrian_ap50 = []
cyclist_ap50 = []

# Populate lists with data
for entry in data['metrics']:
    vehicle_precision.append(entry['precision']['VEHICLE'])
    pedestrian_precision.append(entry['precision']['PEDESTRIAN'])
    cyclist_precision.append(entry['precision']['CYCLIST'])
    
    vehicle_recall.append(entry['recall']['VEHICLE'])
    pedestrian_recall.append(entry['recall']['PEDESTRIAN'])
    cyclist_recall.append(entry['recall']['CYCLIST'])
    
    vehicle_ap50.append(entry['ap50']['VEHICLE'])
    pedestrian_ap50.append(entry['ap50']['PEDESTRIAN'])
    cyclist_ap50.append(entry['ap50']['CYCLIST'])

# Calculate averages
averages = {
    "VEHICLE": {
        "precision": np.mean(vehicle_precision),
        "recall": np.mean(vehicle_recall),
        "ap50": np.mean(vehicle_ap50)
    },
    "PEDESTRIAN": {
        "precision": np.mean(pedestrian_precision),
        "recall": np.mean(pedestrian_recall),
        "ap50": np.mean(pedestrian_ap50)
    },
    "CYCLIST": {
        "precision": np.mean(cyclist_precision),
        "recall": np.mean(cyclist_recall),
        "ap50": np.mean(cyclist_ap50)
    }
}

import matplotlib.pyplot as plt

# Define number of entries for the x-axis
num_entries = len(data['metrics'])
x_values = list(range(1, num_entries + 1))
colors = plt.get_cmap("Set2")
# Replotting with "Epoch" as the X-axis label
fig, axs = plt.subplots(3, 1, figsize=(12, 18))
fig.suptitle("Evolution of Precision, Recall, and AP50 by class in FasterR-CNN training for NuScenes, pretrained", fontsize=14)

# Precision Plot
axs[0].plot(x_values, vehicle_precision, label="VEHICLE", marker='o', color=colors(0))
axs[0].plot(x_values, pedestrian_precision, label="PEDESTRIAN", marker='o', color=colors(1))
axs[0].plot(x_values, cyclist_precision, label="CYCLIST", marker='o', color=colors(2))
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Precision")
axs[0].legend()
axs[0].grid(True)

# Recall Plot
axs[1].plot(x_values, vehicle_recall, label="VEHICLE", marker='o', color=colors(0))
axs[1].plot(x_values, pedestrian_recall, label="PEDESTRIAN", marker='o', color=colors(1))
axs[1].plot(x_values, cyclist_recall, label="CYCLIST", marker='o', color=colors(2))
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Recall")
axs[1].legend()
axs[1].grid(True)

# AP50 Plot
axs[2].plot(x_values, vehicle_ap50, label="VEHICLE", marker='o', color=colors(0))
axs[2].plot(x_values, pedestrian_ap50, label="PEDESTRIAN", marker='o', color=colors(1))
axs[2].plot(x_values, cyclist_ap50, label="CYCLIST", marker='o', color=colors(2))
axs[2].set_xlabel("Epoch")
axs[2].set_ylabel("AP50")
axs[2].legend()
axs[2].grid(True)

# Display plot
plt.savefig("NuScenes_FasterRCNN/results.png")
