from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import string


def add_labels_adjusted(image_name, num_columns, num_rows=1):
###    fig, ax = plt.subplots(figsize=(10, 5)) 
    image = Image.open(image_name+".png")
    image_array = np.array(image)
    dpi=200
    figsize = (image_array.shape[1] / dpi, image_array.shape[0] / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(image_array, interpolation='none')  
    
    # Calculate the spacing based on the number of columns and rows
    column_width = image_array.shape[1] // num_columns
    row_height = image_array.shape[0] // num_rows

    # Define labels based on the number of subfigures
    labels = list(string.ascii_uppercase[:num_columns * num_rows])

    # Generate positions for each label based on the upper left of each subplot
    positions = [(column_width * i + 10, 4 + row_height * (i // num_columns)) for i in range(num_columns * num_rows)]
    
    # Place labels on the plot
    for label, position in zip(labels, positions):
        ax.text(position[0], position[1], label, color='black', fontsize=8, fontweight='bold', va='top', ha='left')

    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(image_name+"_labeled.png", bbox_inches='tight', pad_inches=0, dpi=dpi) 

add_labels_adjusted("figs/gpt4facts_barchart", num_columns=2, num_rows=1)
