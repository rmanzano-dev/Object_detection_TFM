import matplotlib.pyplot as plt

# Provided metrics
precision = 0.5008  # Overall precision
recall = 0.7943  # Overall recall
map_score = 0.3512  # Overall mAP score

# Class-specific mAP scores
class_map = [0.4876, 0.3639, 0.2020]
class_names = ["vehicle", "pedestrian", "cyclist"]

# Plot the overall precision-recall point
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, 'o', markersize=10, label=f"Overall (AP={map_score:.2f})", color='black')

# Plot each class-specific AP score as a distinct point on the PR plot
for idx, (ap, cls) in enumerate(zip(class_map, class_names)):
    plt.plot([], [], 'o', label=f"{cls} (AP={ap:.2f})")  # Plotting for legend purposes

# Annotate the precision and recall values
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Snapshot")
plt.legend()
plt.grid()
plt.show()
