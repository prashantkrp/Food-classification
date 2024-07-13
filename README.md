Food Classification Based on Calories
This project classifies different types of food based on their calorie content. The foods classified in this project are 'apple_pie', 'pizza', and 'salad'. The classification is performed using a Convolutional Neural Network (CNN) model built with TensorFlow and Keras.

Project Structure
The project consists of scripts for training the model and for final testing and classification of food images.

Datasets
The dataset includes images of three types of food:
Apple Pie
Pizza
Salad

Libraries Used For Training:

TensorFlow
TensorFlow Keras
NumPy
Pillow (PIL)
For Final Testing:
OpenCV
NumPy
Pillow (PIL)
TensorFlow Keras

File Descriptions:
train_model.py: Script to train the food classification model. It processes the dataset, fine-tunes the ResNet50 model, and saves the trained model.
food_model.h5: The file where the trained food classification model is saved.
test_model.py: Script to test and run the food classification model. It loads the trained model, processes new images, and predicts their classes.

How It Works:
Data Preprocessing: The dataset is augmented and preprocessed using ImageDataGenerator.
Model Building: ResNet50, a pre-trained CNN model, is used as the base model. Additional layers are added to tailor it for the food classification task.
Model Training: The model is trained on the dataset of food images.
Prediction: The trained model is used to predict the class of new food images based on their calorie content.

Acknowledgements:
TensorFlow: An open-source platform for machine learning.
Keras: A deep learning API written in Python, running on top of TensorFlow.
OpenCV: An open-source computer vision and machine learning software library.
NumPy: A fundamental package for scientific computing with Python.
Pillow (PIL): A Python Imaging Library that adds image processing capabilities to your Python interpreter.
