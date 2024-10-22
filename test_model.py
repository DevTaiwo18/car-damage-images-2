from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('car_damage_model.keras')

# Path to the image for testing
img_path = r'C:\Users\Adeyemi Taiwo\Desktop\car-damage-dataset-2\images\test 1.jpeg'
 # Update with any available image from your dataset

# Load and preprocess the test image
img = Image.open(img_path)
img = img.resize((150, 150))
img = np.array(img)
img = np.expand_dims(img, axis=0)  # Expand dimensions to match model input

# Make predictions
predictions = model.predict(img)
predicted_num_dents = int(round(predictions['num_dents_output'][0][0]))  # Round the number of dents
predicted_size_class = np.argmax(predictions['size_class_output'][0])  # Get index of the highest score

# Map predicted size class to corresponding labels
size_class_map = {0: 'Dime', 1: 'Nickel', 2: 'Quarter', 3: 'Half Dollar'}
predicted_size_class_label = size_class_map[predicted_size_class]

# Output the predictions
print(f"Predicted Number of Dents: {predicted_num_dents}")
print(f"Predicted Size Class: {predicted_size_class_label}")
