import numpy as np
import cv2
import string
import os
from keras.models import load_model
from datetime import datetime
import time

# Define constants
model_path = r"Train model name" #Trained model name goes here
directory_path = r"Checking directory path goes here"  # Path to the directory containing images
imgshape = (50, 200, 1)
character = string.ascii_lowercase + "0123456789"
nchar = len(character)

# Load the trained model
model = load_model(model_path)
print(f"Model loaded from {model_path}")

# Predict function
def predict(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Image not detected for file: {os.path.basename(filepath)}")
        return None
    
    # Resize image to (50, 200)
    img = cv2.resize(img, (200, 50))
    
    # Normalize image
    img = img / 255.0
    
    # Reshape to (50, 200, 1)
    img = np.reshape(img, imgshape)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Predict
    res = model.predict(img)
    
    # Process results
    result = np.reshape(res, (5, len(character)))
    k_ind = [np.argmax(i) for i in result]

    capt = ''.join([character[k] for k in k_ind])
    return capt

# Check all images in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".png") or filename.endswith(".jpg"):  # Consider only image files
        file_path = os.path.join(directory_path, filename)
        
        start_time = time.time()  # Start the timer
        prediction = predict(file_path)
        end_time = time.time()  # End the timer
        
        if prediction:
            # Get the current date and time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Calculate the response time
            response_time = end_time - start_time
            
            # Print the result with the current date, time, and response time
            print(f"{current_time} - Image: {filename} - Predicted Captcha: {prediction} - Response Time: {response_time:.4f} seconds")
