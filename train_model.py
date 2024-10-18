import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.utils import class_weight
from load_dataset import load_dataset
from tensorflow.keras.optimizers import Adam

# Load dataset
images, labels = load_dataset()

# Split labels into two parts: number of dents and size class
num_dents_labels = labels[:, 0]  # Number of dents
size_class_labels = labels[:, 1]  # Size class (Nickel, Dime, etc.)

# Normalize image data (pixel values scaled to [0, 1])
images = images / 255.0

# Calculate class weights for size class to handle class imbalance
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(size_class_labels), y=size_class_labels)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Create sample weights for the size class output
sample_weights_size_class = np.array([class_weights_dict[label] for label in size_class_labels])

# Since `num_dents_output` is regression, we can assign uniform sample weights for the dents output
sample_weights_num_dents = np.ones_like(num_dents_labels)

# Combine the sample weights into a dictionary for both outputs
sample_weights = {
    'num_dents_output': sample_weights_num_dents,
    'size_class_output': sample_weights_size_class
}

# Define the model architecture
def create_model():
    inputs = layers.Input(shape=(150, 150, 3))

    # Add convolutional layers
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Outputs
    num_dents_output = layers.Dense(1, name='num_dents_output')(x)  # Regression for number of dents
    size_class_output = layers.Dense(4, activation='softmax', name='size_class_output')(x)  # Classification for size class

    # Create the model
    model = models.Model(inputs=inputs, outputs={'num_dents_output': num_dents_output, 'size_class_output': size_class_output})

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss={'num_dents_output': 'mean_squared_error', 'size_class_output': 'sparse_categorical_crossentropy'},
                  metrics={'num_dents_output': 'mae', 'size_class_output': 'accuracy'})

    return model

# Create the model
model = create_model()

# Train the model with sample weights for both outputs
history = model.fit(images, 
                    {'num_dents_output': num_dents_labels, 'size_class_output': size_class_labels},
                    epochs=30, batch_size=8, validation_split=0.2,
                    sample_weight=sample_weights)  # Providing sample weights for both outputs

# Save the model in Keras format
model.save('car_damage_model.keras')

# Print the model summary
model.summary()
