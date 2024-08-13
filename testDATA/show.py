from scipy import signal
import numpy as np
from matplotlib import pyplot as plt

# x = np.random.rand(500)
# x = np.arange(2000)
# x = np.load('train300m30.npy')

strLIST = []
for i in np.arange(1, 100+1):
    strLIST.append('test500ml{}.npy'.format(i) )

# for i in [1, 2, 4, 5, 8, 10]:
#     # y = signal.resample_poly(x,1,2)
#     y = abs(np.fft.fft(x[::i]) )
#     y = y[1:int(len(y)/2)]
    
#     plt.figure(1, dpi = 200)
#     plt.plot(x)
#     plt.show()

for i in strLIST:
    plt.figure(1, dpi = 200)
    x = np.load(i)
    x = abs(np.fft.fft(x))
    x = x[1:400]
    plt.plot(x)
    plt.title(i)
    plt.show()



