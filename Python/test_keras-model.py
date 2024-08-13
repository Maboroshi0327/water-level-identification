import tensorflow as tf
models = tf.keras.models

from My_Functions import test_data


sampleRate = 20000
x_test, y_test = test_data(sampleRate)

model = models.load_model('./models/20000_3_60_40_30_relu_0.3.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# model = models.load_model('water_ann.h5')
# model.summary()
# testDATA = np.load('temp.npy')
# # testDATA = np.load('./testDATA/test500ml25.npy')
# testDATA = signal.resample_poly(testDATA, 50000, 50000)
# testDATA = np.abs(fft(testDATA))[1:1250]
# testDATA = Min_Max_Normalization(testDATA, 0, 1)
# testDATA = np.reshape(testDATA, (1, 1249))
# result = model.predict(testDATA)
# print(result)
