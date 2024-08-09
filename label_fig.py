from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
image = Image.open("figs/combined_heatmaps.png")
image_array = np.array(image)

# Function to add labels
def add_labels_adjusted(image_array):
    fig, ax = plt.subplots()
    ax.imshow(image_array)
    
    # Define positions and labels
    labels = ['A', 'B', 'C', 'D']
    positions = [(0, 0), (image_array.shape[1] // 2 + 10, 10), (10, image_array.shape[0] // 2 + 10), (image_array.shape[1] // 2 + 10, image_array.shape[0] // 2 + 10)]
    
    for label, position in zip(labels, positions):
        ax.text(position[0], position[1], label, color='black', fontsize=12, fontweight='bold', va='top', ha='left')
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("figs/combined_heatmaps_labeled.png", bbox_inches='tight', pad_inches=0)

# Apply the function
add_labels_adjusted(image_array)
