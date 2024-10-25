import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Example data: the sums for each class in training and validation sets
class_names = ['Vehículo', 'Peatón', 'Ciclista']  # Adjust based on your classes
train_sums = [106144, 35242, 394]  # Sums for training set
val_sums = [20289, 9017, 254]     # Sums for validation set

# Create a DataFrame with the class distribution for both sets
data = {
    'Clase': class_names * 2,  # Repeat the class names for both sets
    'Total': train_sums + val_sums,  # Combine train and validation counts
    'Conjunto': ['Train'] * len(train_sums) + ['Validation'] * len(val_sums)  # Specify Train and Validation
}

df = pd.DataFrame(data)

# Plot the distribution
plt.figure(figsize=(10, 6))
sns.barplot(x='Clase', y='Total', hue='Conjunto', data=df, palette='Set2')
plt.title('Class Distribution in Train and Validation Sets')
plt.xlabel('Clase')
plt.ylabel('Total')

plt.savefig("img/Waymo_Clases")
