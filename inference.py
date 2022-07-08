import gradio as gr
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import yaml
import albumentations as A

config = yaml.safe_load(open('config.yaml', 'r'))
interpreter = tf.lite.Interpreter(model_path=config['tflite_path'])
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

encoder = LabelBinarizer()

labels = pd.read_csv(config['dataset']['csv_dir'] + 'full.csv').label
labels = np.asarray(labels)
labels = encoder.fit_transform(labels)

transformation = A.Compose(
    [
        A.Rotate(limit=15, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussianBlur(blur_limit=(3, 9), p=0.6),
        A.RandomBrightnessContrast(p=0.4),
    ]
)


def resize(image, image_size=config['img_shape'][0]):
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)


def augment(image):
    return transformation(image=image)['image']


def normalize(image):
    image = image / 255
    return image


def preprocess(image):
    image = augment(image)
    image = resize(image)
    image = normalize(image)
    return image


def inference(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    preds = []
    for _ in range(3):
        input_data = preprocess(image)
        input_data = np.expand_dims(input_data, axis=0)
        input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])
        preds.append(output)

    pred = (preds[0] + preds[1] + preds[2]) / 3
    # res = np.argmax(output)
    pred = encoder.inverse_transform(np.array(pred))
    return pred


interface = gr.Interface(fn=inference, inputs='image',
                         outputs='text', title='Portugese Meals Classification')
interface.launch(share=True, inbrowser=True, server_port=5000)
interface.block_thread()
interface.close()
