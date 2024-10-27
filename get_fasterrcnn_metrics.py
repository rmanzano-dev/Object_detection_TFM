import os
import json
import matplotlib.pyplot as plt

# Folder where JSON files are stored
folder_path = "Waymo_FasterRCNNPTFinal_results_PT"

# Lists to store metric values
epochs = []
map_values = []
map_50_values = []
precision_values = []
recall_values = []
datas = {}
# Load each JSON file in the folder
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith("_json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as f:
            data = json.load(f)
            epoch = int(file_path.split("-")[1].split("_")[0])
            datas[epoch] = data

datas = dict(sorted(datas.items()))
for epoch, values in datas.items():
    # Extract data
    epochs.append(epoch)  # Assumes sequential if "epoch" is missing
    map_values.append(values["map"])
    map_50_values.append(values["map_50"])
    precision_values.append(values["precision"])
    recall_values.append(values["recall"])
# Plot each metric across epochs
plt.figure(figsize=(10, 6))

plt.plot(epochs, map_values, label="mAP5090", marker='o')
plt.plot(epochs, map_50_values, label="mAP_50", marker='o')
plt.plot(epochs, precision_values, label="Precision", marker='o')
plt.plot(epochs, recall_values, label="Recall", marker='o')

dataset = folder_path.split("_")[0]
# Customize plot
plt.title(f"Evolución de las métricas por época en {dataset}, Pre-entrenado")
plt.xlabel("Epochs")
plt.ylabel("Metric Values")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.savefig(f"Metrics_{dataset}_PT")

