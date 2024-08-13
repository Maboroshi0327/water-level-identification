# Full integer quantization

import tensorflow as tf
import numpy as np

from My_Functions import train_data


def data():
    sampleRate = 20000
    x_train , _ = train_data(sampleRate)
    x_train = np.float32(x_train)
    return x_train


def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(data()).batch(1).take(100):
        yield [input_value]


if __name__ == '__main__':
    models = tf.keras.models
    model = models.load_model('./models/20000_3_60_40_30_relu_0.3.h5')
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model_quant = converter.convert()

    with open('./models/quantization_8bits.tflite', 'wb') as f:
        f.write(tflite_model_quant)
