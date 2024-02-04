import numpy as np

# Create a few sample images
image1 = np.random.randint(0, 256, (3, 3))
image2 = np.random.randint(0, 256, (3, 3))
# ... create more images if needed
print("image1 is:" , image1)
print("image2 is:" , image2)
# Label the images and store them in a dictionary
# Ensure all entries are actual images or remove the ellipsis
dataset = {
    'Category_A': [image1],  # Add more images as needed
    'Category_B': [image2]   # Add more images as needed
}

def normalize(image):
    return image / 255

def flatten(image):
    return image.flatten()

print("The dataset is:", dataset)

processed_data = []
for label, images in dataset.items():
    for img in images:
        normalized_img = normalize(img)
        flattened_img = flatten(normalized_img)
        processed_data.append((flattened_img, label))

print("The processed data is:", processed_data)
