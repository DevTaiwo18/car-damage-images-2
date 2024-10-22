import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from load_dataset import load_dataset
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load dataset
images, labels = load_dataset()

# Split labels into two parts: number of dents and size class
num_dents_labels = labels[:, 0]  # Number of dents
size_class_labels = labels[:, 1]  # Size class (Nickel, Dime, etc.)

# Normalize image data (pixel values scaled to [0, 1])
images = images / 255.0

# Calculate class weights for size class to handle class imbalance
class_weights = class_weight.compute_class_weight(class_weight='balanced', 
                                                  classes=np.unique(size_class_labels), 
                                                  y=size_class_labels)
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

# Define the model architecture with regularization
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
    
    # Adding regularization to the dense layer
    x = layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu')(x)
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

# Learning rate scheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Option 1: Train the model directly (without augmentation)
history = model.fit(images, 
                    {'num_dents_output': num_dents_labels, 'size_class_output': size_class_labels},
                    epochs=30, batch_size=8, validation_split=0.2,
                    sample_weight=sample_weights, 
                    callbacks=[lr_scheduler, early_stopping])

# Option 2: Train with data augmentation (optional)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize the ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=20, 
                             width_shift_range=0.2, 
                             height_shift_range=0.2, 
                             shear_range=0.2, 
                             zoom_range=0.2, 
                             horizontal_flip=True, 
                             fill_mode='nearest')

# Augment the data during training (if desired)
# history = model.fit(datagen.flow(images, {'num_dents_output': num_dents_labels, 'size_class_output': size_class_labels}, batch_size=8),
#                     epochs=30,
#                     validation_data=(images, {'num_dents_output': num_dents_labels, 'size_class_output': size_class_labels}),
#                     sample_weight=sample_weights,
#                     callbacks=[lr_scheduler, early_stopping])

# Save the model in Keras format
model.save('car_damage_model.keras')

# Print the model summary
model.summary()

# --- Visualization of Training & Validation Metrics ---
def plot_metrics(history):
    # Plot loss for number of dents
    plt.plot(history.history['num_dents_output_loss'], label='Train Loss (Dents)')
    plt.plot(history.history['val_num_dents_output_loss'], label='Val Loss (Dents)')
    plt.title('Number of Dents - Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()

    # Plot accuracy for size class prediction
    plt.plot(history.history['size_class_output_accuracy'], label='Train Accuracy (Size Class)')
    plt.plot(history.history['val_size_class_output_accuracy'], label='Val Accuracy (Size Class)')
    plt.title('Size Class - Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

# Call the plotting function after training
plot_metrics(history)

# --- Confusion Matrix for Classification Task (Size Class) ---
y_pred_size_class = model.predict(images)['size_class_output']
y_pred_size_class = np.argmax(y_pred_size_class, axis=1)  # Convert probabilities to class labels

# Generate confusion matrix
conf_matrix = confusion_matrix(size_class_labels, y_pred_size_class)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Nickel', 'Dime', 'Quarter', 'Other'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Size Class Prediction")
plt.show()
