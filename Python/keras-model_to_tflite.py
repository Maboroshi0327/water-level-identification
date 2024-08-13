# Dynamic range quantization

import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    models = tf.keras.models
    model = models.load_model('./models/30000_3_60_50_10_tanh_0.3.h5')
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model_quant = converter.convert()

    with open('./models/quantization.tflite', 'wb') as f:
        f.write(tflite_model_quant)
