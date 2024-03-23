import tensorflow as tf
from PIL import Image
import numpy as np
import pickle

# Load your trained model
model = tf.keras.models.load_model(r'C:\Users\Med\Desktop\STRI\Tensorflow\model')


def test_image(image_path, img_width, img_height):
    # Load and preprocess the image you want to test
    image = Image.open(image_path)
    image = image.resize((img_width, img_height))  # Resize the image to match input size
    image = np.array(image) / 255.0  # Normalize pixel values

    # Reshape the image to match the input shape expected by your model
    image = image.reshape(1, img_width, img_height, 3)  # Assuming RGB image

    # Make predictions
    predictions = model.predict(image)

    # Print the predicted class
    if predictions[0][0] > 0.5:
        return "It's a dog!"
    else:
        return "It's a cat!"

if __name__ == "__main__":
    # Load the training history
    with open('training_history.pkl', 'rb') as f:
        history = pickle.load(f)

    # Repeat testing
    image_path = 'cat.jpg'
    img_width, img_height = 150, 150  # Define image dimensions
    for _ in range(10):  # Repeat testing 10 times
        result = test_image(image_path, img_width, img_height)
        print(result)
