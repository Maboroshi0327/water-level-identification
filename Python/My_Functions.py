import numpy as np
from numpy.fft import fft
from scipy import signal

# -------------------------------------------------------------------------------------------------
# 最大值最小值正規化
def Min_Max_Normalization(array, min=0, max=1):    # min,max=輸出資料大小的範圍
    array = array.flatten()
    array_max = np.max(array)
    array_min = np.min(array)
    ratio = (max - min) / (array_max - array_min)
    return min + ratio * (array - array_min)


# -------------------------------------------------------------------------------------------------
# Data
def train_data(sampleRate):
    x_train = np.zeros((500, 283))
    y_train = np.zeros((500, 5))
    for i in np.arange(0, 100):
        list100 = np.load('./trainDATA/train100ml{}.npy'.format(i + 1))
        list200 = np.load('./trainDATA/train200ml{}.npy'.format(i + 1))
        list300 = np.load('./trainDATA/train300ml{}.npy'.format(i + 1))
        list400 = np.load('./trainDATA/train400ml{}.npy'.format(i + 1))
        list500 = np.load('./trainDATA/train500ml{}.npy'.format(i + 1))
        list100 = signal.resample_poly(list100, sampleRate, 50000)
        list200 = signal.resample_poly(list200, sampleRate, 50000)
        list300 = signal.resample_poly(list300, sampleRate, 50000)
        list400 = signal.resample_poly(list400, sampleRate, 50000)
        list500 = signal.resample_poly(list500, sampleRate, 50000)
        fft100 = np.abs(fft(list100))[18:301]
        fft200 = np.abs(fft(list200))[18:301]
        fft300 = np.abs(fft(list300))[18:301]
        fft400 = np.abs(fft(list400))[18:301]
        fft500 = np.abs(fft(list500))[18:301]
        x_train[i*5] = fft100
        x_train[i*5 + 1] = fft200
        x_train[i*5 + 2] = fft300
        x_train[i*5 + 3] = fft400
        x_train[i*5 + 4] = fft500
        y_train[i*5] = [1, 0, 0, 0, 0]
        y_train[i*5 + 1] = [0, 1, 0, 0, 0]
        y_train[i*5 + 2] = [0, 0, 1, 0, 0]
        y_train[i*5 + 3] = [0, 0, 0, 1, 0]
        y_train[i*5 + 4] = [0, 0, 0, 0, 1]
    for index in np.arange(x_train.shape[0]):
        x_train[index] = Min_Max_Normalization(x_train[index], min=0, max=1)
    return x_train, y_train


def test_data(sampleRate):
    x_test = np.zeros((500, 283))
    y_test = np.zeros((500, 5))
    for i in np.arange(0, 100):
        list100 = np.load('./testDATA/test100ml{}.npy'.format(i + 1))
        list200 = np.load('./testDATA/test200ml{}.npy'.format(i + 1))
        list300 = np.load('./testDATA/test300ml{}.npy'.format(i + 1))
        list400 = np.load('./testDATA/test400ml{}.npy'.format(i + 1))
        list500 = np.load('./testDATA/test500ml{}.npy'.format(i + 1))
        list100 = signal.resample_poly(list100, sampleRate, 50000)
        list200 = signal.resample_poly(list200, sampleRate, 50000)
        list300 = signal.resample_poly(list300, sampleRate, 50000)
        list400 = signal.resample_poly(list400, sampleRate, 50000)
        list500 = signal.resample_poly(list500, sampleRate, 50000)
        fft100 = np.abs(fft(list100))[18:301]
        fft200 = np.abs(fft(list200))[18:301]
        fft300 = np.abs(fft(list300))[18:301]
        fft400 = np.abs(fft(list400))[18:301]
        fft500 = np.abs(fft(list500))[18:301]
        x_test[i*5] = fft100
        x_test[i*5 + 1] = fft200
        x_test[i*5 + 2] = fft300
        x_test[i*5 + 3] = fft400
        x_test[i*5 + 4] = fft500
        y_test[i*5] = [1, 0, 0, 0, 0]
        y_test[i*5 + 1] = [0, 1, 0, 0, 0]
        y_test[i*5 + 2] = [0, 0, 1, 0, 0]
        y_test[i*5 + 3] = [0, 0, 0, 1, 0]
        y_test[i*5 + 4] = [0, 0, 0, 0, 1]
    for index in np.arange(x_test.shape[0]):
        x_test[index] = Min_Max_Normalization(x_test[index], min=0, max=1)
    return x_test, y_test

if __name__ == "__main__":
    x_train, y_train = train_data(30000)
    x_test, y_test = test_data(30000)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)