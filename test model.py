import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('C:\\Users\\Prashant kumar\\PycharmProjects\\pythonProject9\\food_classification_model.keras')

# Predefined mapping of food items to calories (you'll need to fill this with actual data)
food_calories_mapping = {
    'apple_pie': 300,
    'pizza': 270,
    'salad': 150,
    # Add more items as needed
}

# Load Haar Cascade classifiers for face and full-body detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
fullbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')


def preprocess_image(image):
    """
    Preprocess the image to match the input requirements of the model.
    """
    img = Image.fromarray(image)
    img = img.resize((224, 224))  # Assuming the model expects 224x224 input images
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def detect_human(image):
    """
    Detect humans in the image using Haar Cascade classifiers.
    Returns True if humans are detected, False otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detect full bodies
    bodies = fullbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If faces or bodies are detected, return True
    if len(faces)>0 or len(bodies)>0:
        return True
    else:
        return False


def predict_and_estimate_calories(image):
    """
    Predict the food item from the image and estimate the calories.
    """
    try:
        # Check for humans in the image
        if detect_human(image):
            print("Human detected, skipping food classification.")
            return None, None

        # Preprocess the image
        img_array = preprocess_image(image)

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]

        # Get the predicted class label (assuming you have a class index to label mapping)
        class_labels = list(food_calories_mapping.keys())  # List of class labels
        predicted_class = class_labels[predicted_class_idx]

        # Estimate calories
        calories = food_calories_mapping.get(predicted_class, "Unknown")

        return predicted_class, calories
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def live_food_detection():
    """
    Perform live food detection using the camera.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Predict and estimate calories only if no human detected
        food_item, calories = predict_and_estimate_calories(frame)

        # Display the results on the frame
        if food_item and calories:
            cv2.putText(frame, f'Food: {food_item}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Calories: {calories}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Live Food Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


# Start live food detection
live_food_detection()
