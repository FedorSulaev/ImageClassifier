import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json
from PIL import Image
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


def process_image(image):
    image_tensor = tf.convert_to_tensor(image)
    image_tensor = tf.image.resize(
        image_tensor, tf.Variable([224, 224], tf.int32))
    image_tensor /= 255
    return image_tensor.numpy()


def predict(image_path, model, top_k):
    image = np.asarray(Image.open(image_path))
    processed_image = process_image(image)
    predictions = model.predict(np.expand_dims(processed_image, 0))
    top_predictions = tf.nn.top_k(predictions, top_k)
    to_str = np.vectorize(str)
    return (top_predictions.values.numpy().squeeze(), to_str(top_predictions.indices.numpy().squeeze()))


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('image_path', action='store')
parser.add_argument('model_path', action='store')
parser.add_argument('--top_k', dest='top_k',
                    action='store', type=int, default=5)
parser.add_argument('--category_names',
                    dest='category_names_path', action='store')

arguments = parser.parse_args()

# print(arguments.image_path)
# print(arguments.model_path)
# print(arguments.top_k)
# print(arguments.category_names_path)

# Load model
model = tf.keras.models.load_model(
    arguments.model_path, custom_objects={'KerasLayer': hub.KerasLayer})

# Load category labels
with open(arguments.category_names_path, 'r') as f:
    class_names = json.load(f)

# Make prediction
probs, classes = predict(arguments.image_path, model, arguments.top_k)

# Output results
for i in range(len(probs)):
    print(str.format("Class: {0}, probability: {1}",
                     class_names[classes[i]], probs[i]))
