import numpy as np 
import matplotlib.pyplot as plt
import os
from keras import layers
from keras.models import Model
import cv2
import string

# Path to your dataset
data_path = r"Give the data path for training here"  # Change this to your actual path

# Define the image size
imgshape = (50, 200, 1)

# All symbols captcha can contain
character = string.ascii_lowercase + "0123456789"
nchar = len(character)

# Preprocess image function
def preprocess(data_path):
    n = len(os.listdir(data_path))
    X = np.zeros((n, 50, 200, 1))  # Assuming a fixed image size of 50x200 with 1 channel
    y = np.zeros((5, n, len(character)))  # Assuming 5 characters in the captcha

    for i, pic in enumerate(os.listdir(data_path)):
        img = cv2.imread(os.path.join(data_path, pic), cv2.IMREAD_GRAYSCALE)
        pic_target = pic[:-4]  # Remove file extension

        # Ensure pic_target has exactly 5 characters
        if len(pic_target) > 5:
            pic_target = pic_target[:5]  # Truncate to 5 characters
        elif len(pic_target) < 5:
            pic_target = pic_target.ljust(5, '_')  # Pad with underscores if less than 5 characters

        if img is None or img.shape[0] != 50 or img.shape[1] != 200:
            img = cv2.resize(img, (200, 50))  # Resize to 200x50 if necessary

        img = img / 255.0  # Normalize image
        img = np.reshape(img, (50, 200, 1))  # Reshape to (50, 200, 1)

        target = np.zeros((5, len(character)))

        for j, k in enumerate(pic_target):
            index = character.find(k)
            if index != -1:
                target[j, index] = 1

        X[i] = img
        y[:, i] = target

    return X, y

# Load and preprocess data
X, y = preprocess(data_path)

# Ensure there are at least 20,030 images in the dataset
if len(X) < 20030:
    raise ValueError(f"Dataset contains only {len(X)} images. Please add more images to reach 20,030.")

# Split the data: Use 20,030 images for training and the rest for testing
X_train, y_train = X[:20030], y[:, :20030]
X_test, y_test = X[20030:], y[:, 20030:]

# Create model function
def createmodel():
    img = layers.Input(shape=imgshape)
    conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(img)
    mp1 = layers.MaxPooling2D(padding='same')(conv1)
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
    mp2 = layers.MaxPooling2D(padding='same')(conv2)
    conv3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp2)
    bn = layers.BatchNormalization()(conv3)
    mp3 = layers.MaxPooling2D(padding='same')(bn)

    flat = layers.Flatten()(mp3)
    outs = []
    for _ in range(5):
        dens1 = layers.Dense(64, activation='relu')(flat)
        drop = layers.Dropout(0.5)(dens1)
        res = layers.Dense(nchar, activation='sigmoid')(drop)
        outs.append(res)

    model = Model(img, outs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'] * 5  # Using the same metric for all 5 outputs
    )

    return model

# Create and summarize the model
model = createmodel()
model.summary()

# Train the model
hist = model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], 
                 batch_size=32, epochs=60, validation_split=0.2)

# Save the model
model_save_path = 'captcha_model_20030.h5'
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Plot graphs for loss and accuracy
def plot_history(hist):
    # Print available keys
    print(hist.history.keys())
    
    # Iterate over all dense layers
    for i in range(5):
        acc_key = f'dense_{i}_accuracy'
        val_acc_key = f'val_dense_{i}_accuracy'
        
        # Check if keys exist before plotting
        if acc_key in hist.history and val_acc_key in hist.history:
            plt.plot(hist.history[acc_key], label=f'Training Accuracy {i}')
            plt.plot(hist.history[val_acc_key], label=f'Validation Accuracy {i}')
    
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

plot_history(hist)

# Evaluate the model
train_loss = model.evaluate(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]])
print(f"Loss on training set = {train_loss[0]}")

test_loss = model.evaluate(X_test, [y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]])
print(f"Loss on testing set = {test_loss[0]}")

# Predict function
def predict(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Image not detected")
        return None
    
    # Resize image to (50, 200)
    img = cv2.resize(img, (200, 50))
    
    # Normalize image
    img = img / 255.0
    
    # Reshape to (50, 200, 1)
    img = np.reshape(img, (50, 200, 1))
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Predict
    res = model.predict(img)
    
    # Process results
    result = np.reshape(res, (5, len(character)))
    k_ind = [np.argmax(i) for i in result]

    capt = ''.join([character[k] for k in k_ind])
    return capt

# Test the model on a sample image
sample_image_path = 'krgkhdhg.png'  # Replace with the path to a sample image
print(f"Predicted Captcha = {predict(sample_image_path)}")
