import json
from PIL import Image
import numpy as np
import os

# Load the dataset
def load_dataset(dataset_path='dent-dataset.jsonl', images_folder='images/'):
    # Load the JSONL dataset
    with open(dataset_path, 'r') as file:
        dataset = [json.loads(line) for line in file]

    # Prepare arrays to hold image data and labels
    image_data = []
    labels = []

    # Define the size to which all images will be resized
    image_size = (150, 150)

    # Process each entry in the dataset
    for entry in dataset:
        # Load and process the image
        image_path = os.path.join(images_folder, entry['image'])
        image = Image.open(image_path)
        image = image.resize(image_size)  # Resize to (150, 150)
        image_array = np.array(image)

        # Add the image to the dataset
        image_data.append(image_array)

        # Process the response to extract the number of dents and size
        response = entry['response']
        num_dents, size_class = extract_labels(response)

        # Add the labels
        labels.append([num_dents, size_class])

    # Convert lists to numpy arrays
    image_data = np.array(image_data)
    labels = np.array(labels)

    return image_data, labels

# Extract labels from response text
def extract_labels(response):
    # Count the number of dents
    if '1 dent' in response:
        num_dents = 1
    elif '2 dents' in response:
        num_dents = 2
    else:
        num_dents = 0  # Assume no dents mentioned

    # Determine size class (Nickel=1, Quarter=2, Half Dollar=3)
    if 'Nickel' in response:
        size_class = 1
    elif 'Quarter' in response:
        size_class = 2
    elif 'Half Dollar' in response:
        size_class = 3
    else:
        size_class = 0  # No dent or another class

    return num_dents, size_class

if __name__ == "__main__":
    # Load dataset and check shapes
    images, labels = load_dataset()
    print("Loaded dataset:")
    print(f"Images: {images.shape}, Labels: {labels.shape}")
