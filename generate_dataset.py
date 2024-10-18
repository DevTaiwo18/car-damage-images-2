import os
import json
import random

# Base path for the image folder and output file
base_image_folder = 'images'
output_file = 'dent-dataset.jsonl'

# Questions for counting dents and assessing their size
questions = [
    "How many dents are visible in this image?",
    "What is the size of the largest dent in this image? (Dime, Nickel, Quarter, or Half Dollar)"
]

# Responses based on dent count and size
dent_responses = [
    "There is 1 dent visible in the image, and it is approximately the size of a Nickel.",
    "The image shows 3 dents. The largest one is the size of a Quarter, while the other two are smaller, around the size of a Dime.",
    "This image has 2 dents. Both are approximately the size of a Nickel.",
    "There are multiple dents, and the largest one appears to be around the size of a Half Dollar.",
    "There is no visible dent in this image."
]

# Function to get a random dent-related response
def get_assistant_response():
    return random.choice(dent_responses)

# Function to generate the dataset
def generate_dataset(base_image_folder, output_file):
    dataset = []
    
    # List the images from the folder
    image_files = os.listdir(base_image_folder)
    
    # Iterate through each image and generate a dataset entry
    for image in image_files:
        entry = {
            'image': image,
            'questions': questions,
            'response': get_assistant_response()
        }
        dataset.append(entry)
    
    # Write the dataset to a JSONL file
    with open(output_file, 'w') as file:
        for item in dataset:
            file.write(json.dumps(item) + '\n')

# Generate the dataset
generate_dataset(base_image_folder, output_file)
