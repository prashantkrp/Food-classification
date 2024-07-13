import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image  # Import PIL for image processing

# Define your data directory
data_dir = 'C:\\Users\\Prashant kumar\\PycharmProjects\\pythonProject9\\food-101\\images'

# Parameters
batch_size = 32
epochs = 10

# Prepare data generators
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),  # Adjust target size as per your image dimensions
    batch_size=batch_size,
    class_mode='categorical',  # Ensure this matches your dataset
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),  # Adjust target size as per your image dimensions
    batch_size=batch_size,
    class_mode='categorical',  # Ensure this matches your dataset
    subset='validation')

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False)

# Add custom layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(101, activation='softmax')(x)

# Create model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='food_classification_model.keras', save_best_only=True, monitor='val_loss', mode='min'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
]

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=epochs,
    callbacks=callbacks
)

# Map food items to calorie content (example values, you need a real mapping)
calorie_mapping = {
    'apple_pie': 300,
    'baby_back_ribs': 400,
    # Add all 101 food items with their calorie content
}

# Function to predict food item and estimate calories
def predict_and_estimate_calories(image_path):
    img = Image.open(image_path)  # Use PIL to open the image
    img = img.resize((224, 224))  # Resize image to match model's expected sizing
    img_array = np.asarray(img)  # Convert PIL image to numpy array
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = list(train_generator.class_indices.keys())[predicted_class[0]]  # Corrected indexing

    estimated_calories = calorie_mapping.get(predicted_label, "Unknown")

    return predicted_label, estimated_calories

# Example usage
image_path = 'path_to_sample_image.jpg'
food_item, calories = predict_and_estimate_calories(image_path)
print(f"Food Item: {food_item}, Estimated Calories: {calories}")
