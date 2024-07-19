import tensorflow as tf
from tensorflow.keras import models, layers # type: ignore
from tensorflow.keras.models import save_model, load_model # type: ignore
import requests
import json
import numpy as np
from keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array  # type: ignore

# Langflow API details
BASE_API_URL = "http://127.0.0.1:7860/api/v1/run"
FLOW_ID = "0e132174-9868-44ec-ae9e-5672282994a5"
ENDPOINT = ""

# Tweaks for the flow
TWEAKS = {
    "ChatInput-unea8": {},
    "Prompt-xE6wO": {},
    "OllamaModel-UScGQ": {},
    "ChatOutput-GikkG": {}
}

def run_flow(message: str, endpoint: str = FLOW_ID) -> dict:
    """
    Run a flow with a given message.

    :param message: The message to send to the flow
    :param endpoint: The ID or the endpoint name of the flow
    :return: The JSON response from the flow
    """
    api_url = f"{BASE_API_URL}/{endpoint}"
    payload = {
        "input_value": message,
        "output_type": "chat",
        "input_type": "chat",
        "tweaks": TWEAKS
    }
    response = requests.post(api_url, json=payload)
    return response.json()

# Model setup
IMAGE_SIZE = 256
CHANNELS = 3
BATCH_SIZE = 32

#Scaling the dataset
scaling = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1.0/255)
])

#Using data augmentation for better performance
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

#Model
input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
model1 = models.Sequential([
    scaling,
    data_augmentation,
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape = input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])
model1.build(input_shape=input)

# Saving the model
model1.save('model1.keras')

# Loading the model
cnn_model = load_model('model1.keras')

classes = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def get_disease_recommendation(image_path):
    # Loading the image
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    
    # Converting the image to a numpy array
    img_array = img_to_array(img)
    
    # Expanding the dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    # Making the prediction
    disease_prediction = cnn_model.predict(img_array)
    
    disease = classes[np.argmax(disease_prediction[0])]
    print('Predicted disease label by model:', disease)
    
    # Sending the predicted disease name to the Langflow API
    response = run_flow(disease)
    print("Response from Langflow:", json.dumps(response, indent=2))

# Example usage
image_path = "leave1.jpg"
get_disease_recommendation(image_path)