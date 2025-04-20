"""
Object Recognition with Pretrained Model

This script demonstrates how to use a pretrained model for object recognition
instead of training one from scratch. It uses ResNet50 pretrained on ImageNet.
"""

# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
from io import BytesIO
import os

# Set random seed for reproducibility
np.random.seed(42)

def load_image(img_path, target_size=(224, 224)):
    """
    Load and preprocess an image from a file path or URL
    """
    if img_path.startswith(('http://', 'https://')):
        # Load image from URL
        response = requests.get(img_path)
        img = Image.open(BytesIO(response.content))
    else:
        # Load image from file path
        img = Image.open(img_path)
    
    # Resize image to target size
    img = img.resize(target_size)
    
    # Convert image to array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to create batch (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess input for ResNet50
    return preprocess_input(img_array)

def predict_image(model, img_array, top_n=5):
    """
    Make predictions using the model and return top N results
    """
    predictions = model.predict(img_array)
    return decode_predictions(predictions, top=top_n)[0]

def display_prediction(img_path, predictions):
    """
    Display the image and its predictions
    """
    # Load original image for display
    if img_path.startswith(('http://', 'https://')):
        response = requests.get(img_path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(img_path)
    
    # Create figure with two subplots
    plt.figure(figsize=(12, 5))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Input Image')
    
    # Display predictions
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(predictions))
    
    # Extract class names and probabilities
    labels = [pred[1] for pred in predictions]
    scores = [pred[2] for pred in predictions]
    
    # Create horizontal bar chart
    plt.barh(y_pos, scores, align='center')
    plt.yticks(y_pos, labels)
    plt.xlabel('Probability')
    plt.title('Top Predictions')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load pretrained ResNet50 model
    print("Loading pretrained ResNet50 model...")
    model = ResNet50(weights='imagenet')
    print("Model loaded successfully!")
    
    # Example images to test (you can replace these with your own)
    test_images = [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/1920px-Cat_August_2010-4.jpg',  # Cat
        'https://upload.wikimedia.org/wikipedia/commons/6/6a/Eagle_in_flight.jpg',  # Eagle
        'https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/2010_Volkswagen_Golf_VI_–_5-door_hatchback_%282015-07-24%29_01.jpg/1920px-2010_Volkswagen_Golf_VI_–_5-door_hatchback_%282015-07-24%29_01.jpg',  # Car
    ]
    
    # Process each image
    for i, img_path in enumerate(test_images):
        print(f"\nProcessing image {i+1}/{len(test_images)}...")
        
        # Load and preprocess image
        img_array = load_image(img_path)
        
        # Make prediction
        predictions = predict_image(model, img_array)
        
        # Display results
        print("Top predictions:")
        for j, (imagenet_id, label, score) in enumerate(predictions):
            print(f"{j+1}: {label} ({score:.2f})")
        
        # Display image with predictions
        display_prediction(img_path, predictions)

def predict_local_image(model, image_path):
    """
    Function to predict a local image file
    """
    # Load and preprocess image
    img_array = load_image(image_path)
    
    # Make prediction
    predictions = predict_image(model, img_array)
    
    # Display results
    print(f"Predictions for {image_path}:")
    for j, (imagenet_id, label, score) in enumerate(predictions):
        print(f"{j+1}: {label} ({score:.2f})")
    
    # Display image with predictions
    display_prediction(image_path, predictions)

if __name__ == "__main__":
    main()

    # Uncomment the following code to predict on a local file
    """
    # Load pretrained ResNet50 model
    model = ResNet50(weights='imagenet')
    
    # Path to your local image
    local_image_path = "path/to/your/image.jpg"
    
    # Make prediction on local image
    predict_local_image(model, local_image_path)
    """