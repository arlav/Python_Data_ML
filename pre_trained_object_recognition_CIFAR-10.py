"""
CIFAR-10 Object Recognition with Pretrained Model

This script is based on the original notebook but uses a pretrained model
for object recognition on the CIFAR-10 dataset instead of training one.
"""

# Load necessary packages
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD

# Fix random seed for reproducibility
seed = 6
np.random.seed(seed)

def load_and_preprocess_data():
    """Load and preprocess CIFAR-10 dataset"""
    # Load the data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Normalize the inputs from 0-255 to 0.0-1.0
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Convert class labels to one-hot encoding
    Y_train = to_categorical(y_train)
    Y_test = to_categorical(y_test)
    
    return X_train, Y_train, X_test, Y_test

def display_images(images, predictions=None, true_labels=None):
    """Display a grid of images with predictions and true labels if provided"""
    # Create a mapping from class indices to class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Create a grid of 3x3 images
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    fig.subplots_adjust(hspace=0.5)
    axs = axs.flatten()
    
    for i, img in enumerate(images[:9]):  # Display up to 9 images
        # Set title with predictions if available
        if predictions is not None and true_labels is not None:
            pred_label = class_names[predictions[i]]
            true_label = class_names[true_labels[i]]
            title = f'Pred: {pred_label}\nTrue: {true_label}'
            # Color the title based on whether prediction is correct
            color = 'green' if pred_label == true_label else 'red'
            axs[i].set_title(title, color=color)
        
        # Hide axes
        axs[i].axis('off')
        
        # Display the image
        axs[i].imshow(img)
    
    plt.tight_layout()
    plt.show()

def create_pretrained_model(input_shape, num_classes):
    """Create a model using pretrained weights"""
    # Option 1: Using VGG16 pretrained on ImageNet
    # Note: CIFAR-10 images are 32x32, while VGG16 expects 224x224
    # We'll use a smaller model or adapt the images
    
    # Load the pretrained model without the top classification layers
    base_model = VGG16(weights='imagenet', 
                       include_top=False, 
                       input_shape=input_shape)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add new classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def load_or_create_model(input_shape, num_classes, weights_path=None):
    """Load a model with pretrained weights or create a new one"""
    model = create_pretrained_model(input_shape, num_classes)
    
    # If weights are provided, load them
    if weights_path and os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
    else:
        print("Created model with ImageNet weights")
    
    return model

def main():
    # Load and preprocess data
    X_train, Y_train, X_test, Y_test = load_and_preprocess_data()
    
    print("Training Images:", X_train.shape)
    print("Testing Images:", X_test.shape)
    
    # Display some sample images
    print("Sample images from the dataset:")
    display_images(X_train[:9])
    
    # CIFAR-10 images are 32x32, but pretrained models typically expect larger images
    # We'll resize them to 48x48 to preserve detail while keeping computation manageable
    # (In a real application, you might use 224x224 for better results)
    input_shape = (48, 48, 3)
    num_classes = 10
    
    # Resize images for the pretrained model
    X_train_resized = tf.image.resize(X_train, [48, 48]).numpy()
    X_test_resized = tf.image.resize(X_test, [48, 48]).numpy()
    
    # Create model with pretrained weights
    model = load_or_create_model(input_shape, num_classes)
    
    # Print model summary
    model.summary()
    
    # Fine-tune the model for a few epochs (this is optional)
    # In practice, you might want to skip this if you already have a well-performing model
    print("Fine-tuning the model for 5 epochs...")
    model.fit(X_train_resized, Y_train, 
              validation_data=(X_test_resized, Y_test),
              epochs=5, 
              batch_size=32,
              verbose=1)
    
    # Evaluate the model
    print("Evaluating the model...")
    loss, accuracy = model.evaluate(X_test_resized, Y_test, verbose=1)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Make predictions on test set
    print("Making predictions...")
    predictions = model.predict(X_test_resized)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(Y_test, axis=1)
    
    # Display some test images with predictions
    print("Sample predictions:")
    display_images(X_test[:9], predicted_classes[:9], true_classes[:9])
    
    # You could save the model here if desired
    # model.save('cifar10_pretrained_model.h5')

if __name__ == "__main__":
    main()