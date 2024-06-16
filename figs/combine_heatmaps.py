from PIL import Image

# Paths to your images
image_files = ['heatmap_base_tqa.png', 'heatmap_zerotunedl14attnvo_tqa.png', 'heatmap_negtuned_tqa.png', 'heatmap_lietuned_tqa.png']

# Open the images
images = [Image.open(x) for x in image_files]

# Calculate maximum width and height of images
max_width = max(img.width for img in images)
max_height = max(img.height for img in images)

# New image size should be two images wide by two images tall
new_image = Image.new('RGB', (max_width * 2, max_height * 2))

# Calculate positions to center each image in its grid cell
positions = [
    ((max_width - images[0].width) // 2, (max_height - images[0].height) // 2),
    (max_width + (max_width - images[1].width) // 2, (max_height - images[1].height) // 2),
    ((max_width - images[2].width) // 2, max_height + (max_height - images[2].height) // 2),
    (max_width + (max_width - images[3].width) // 2, max_height + (max_height - images[3].height) // 2)
]

# Paste images into the new image
for img, pos in zip(images, positions):
    new_image.paste(img, pos)

new_image.save('combined_heatmaps.png')
