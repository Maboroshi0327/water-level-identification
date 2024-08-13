import tensorflow as tf
import numpy as np
import time

from My_Functions import test_data


def data():
    sampleRate = 20000
    x_test, y_test = test_data(sampleRate)
    y_test = np.argmax(y_test, axis=1)
    return x_test, y_test


def TFLite_Predict(model_path, testDATA, index):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on testDATA.
    input_shape = input_details[0]['shape']
    testDATA = np.float32(np.reshape(testDATA[index], input_shape))
    interpreter.set_tensor(input_details[0]['index'], testDATA)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


if __name__ == '__main__':
    x_test, y_test = data()
    print(y_test.shape)

    start = time.process_time()
    acc_cnt = 0
    for index in range(500):
        result = TFLite_Predict(model_path='./models/quantization.tflite', testDATA=x_test, index=index)
        result_index = np.argmax(result[0])
        if result_index == y_test[index]:
            acc_cnt += 1
    end = time.process_time()

    print(f'Total Time Spend:{end - start}')
    acc = acc_cnt / 500
    print(f'acc:{acc}')
