import gradio as gr
import tensorflow.lite as tflite
from cv2 import cvtColor, resize, INTER_AREA, COLOR_RGB2BGR
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from yaml import safe_load

config = safe_load(open('config.yaml', 'r'))


def resize_image(image, image_size=config['img_shape'][0]):
    return resize(image, (image_size, image_size), interpolation=INTER_AREA)


def normalize(image):
    image = image / 255
    image = image.astype(np.float32)
    return image


def preprocess(image):
    image = resize_image(image)
    image = normalize(image)
    return image


classes = ['aletria',
           'arroz_cabidela',
           'bacalhau_bras',
           'bacalhau_natas',
           'batatas_fritas',
           'bolo_chocolate',
           'cachorro',
           'caldo_verde',
           'cozido_portuguesa',
           'croissant',
           'donuts',
           'esparguete_bolonhesa',
           'feijoada',
           'francesinha',
           'gelado',
           'hamburguer',
           'jardineira',
           'nata',
           'ovo',
           'pasteis_bacalhau',
           'pizza',
           'tripas_moda_porto',
           'waffles']

classes.sort()

encoder = LabelBinarizer()
classes_oh = np.asarray(classes)
classes_oh = encoder.fit_transform(classes_oh)
del classes_oh

interpreter = tflite.Interpreter(model_path=config['tflite_path'])
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]


def inference(image):
    image = cvtColor(image, COLOR_RGB2BGR)
    image = preprocess(image)
    image = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details['index'])

    # res = np.argmax(output)
    pred = encoder.inverse_transform(np.array(output))
    return pred[0]


interface = gr.Interface(fn=inference, inputs='image',
                         outputs='text', title='Portugese Meals Classification')
interface.launch(share=True, inbrowser=True)
interface.block_thread()
interface.close()
