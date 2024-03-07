import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Function to load data from a JSON file
def load_data(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            try:
                item = json.loads(line)
                if 'loss' in item and 'epoch' in item:
                    data.append(item)
            except json.JSONDecodeError:
                pass
    return data

# Function to plot data with a specified color
def plot_data(data, color, label=None):
    loss_values = [item['loss'] for item in data]
    epochs = [item['epoch'] for item in data]
    plt.plot(epochs, loss_values, marker='o', color=color, label=label)

# Specify the folder containing JSON files
folder_path = './json_loss'

# Load all JSON files in the folder
files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

# Generate a unique color for each JSON file
colors = cm.rainbow(np.linspace(0, 1, len(files)))

# Load and plot data from all JSON files in the folder
for i, file_name in enumerate(files):
    file_path = os.path.join(folder_path, file_name)
    data = load_data(file_path)
    plot_data(data, colors[i], label=file_name)

# Add a legend to the plot
plt.legend()

# Set labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')

# Adjust the size and DPI of the plot
fig = plt.gcf()
fig.set_size_inches(20, 20)  # Adjust the size as desired (width, height)
dpi = 300  # Set the desired DPI value

# Save the plot as an image file
plt.savefig('combined_loss_vs_epoch.png', dpi=dpi)

# Display the plot (optional)
plt.show()
