# Captcha_Model_Training
Code To Train Model For Solving Captchas With Atleast 21k Images, Using CNN

# CAPTCHA Solver Model

This repository contains a CAPTCHA solver model built using Convolutional Neural Networks (CNN). The model is trained on a dataset of at least 21,000 CAPTCHA images and achieves an average accuracy of 98%. It is designed to handle tricky and complex CAPTCHAs, with an average response time of 0.9 seconds.

## Features

- **High Accuracy**: The model consistently achieves an accuracy of 98%.
- **Fast Response**: Processes CAPTCHAs with an average response time of 0.9 seconds.
- **Noise Handling**: Includes an optional utility to remove noise from images, improving performance on noisy CAPTCHA images.

## Files

- **`captcha_solver.py`**: This script trains the CAPTCHA solver model using CNN. It requires a dataset of at least 21,000 CAPTCHA images for effective training.
  
- **`model_check.py`**: After training, use this script to evaluate the model's accuracy and response time.

- **`noise.py`**: An additional utility for removing noise from images. Use this if your CAPTCHA images have significant noise that may affect model performance.

## Usage

1. **Training the Model**:
   - Run `captcha_solver.py` to train the model. Ensure you have a dataset of at least 21,000 CAPTCHA images.

2. **Evaluating the Model**:
   - After training, run `model_check.py` to check the model's accuracy and response time.

3. **Removing Noise**:
   - If your images are noisy, you can use `noise.py` to clean them up before feeding them into the model.

## Requirements

- Python 3.x
- TensorFlow or PyTorch (depending on your implementation)
- Other dependencies as specified in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/captcha-solver.git
