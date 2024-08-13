from My_Functions import Min_Max_Normalization
import serial
import tensorflow as tf
import numpy as np
from numpy.fft import fft
from scipy import signal
from time import sleep
from matplotlib import pyplot as plt
models = tf.keras.models


sleep(1)

COM_PORT = 'COM3'                           # 指定通訊埠名稱
BAUD_RATES = 115200                         # 設定傳輸速率
ser = serial.Serial(COM_PORT, BAUD_RATES)   # 初始化序列通訊埠

fs = 50000
times = 0.25  # sec
lens1 = int(times * fs)
lens2 = int(lens1 * 2)

b1 = bytes()
list1 = np.zeros(lens1)

ser.write(b'A')

while True:
    if ser.in_waiting:
        data_raw = ser.read_all()
        b1 = b1 + data_raw
        if len(b1) == lens2:
            ser.close()
            print('OK')
            for i in range(0, lens2, 2):
                unpackBytes = int.from_bytes(b1[i:i+2], byteorder='little', signed=False)
                index = int(i/2)
                list1[index] = unpackBytes
            break

model = models.load_model('water_ann_10_3_40_40_30_relu_0.2.h5')
model.summary()
testDATA = list1
testDATA = signal.resample_poly(testDATA, 10000, 50000)
testDATA = np.abs(fft(testDATA))[1:1250]
testDATA = Min_Max_Normalization(testDATA, 0, 1)
testDATA = np.reshape(testDATA, (1, 1249))
result = model.predict(testDATA)[0]

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
x = ['100', '200', '300', '400', '500']
plt.figure(1)
plt.bar(x, result)
plt.xlabel('水位(ml)')
plt.ylabel('信心值')
plt.show()