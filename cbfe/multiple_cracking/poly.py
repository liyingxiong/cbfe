'''
Created on 04.04.2017

@author: Yingxiong
'''
import os
import numpy as np
import matplotlib.pyplot as plt

test_file_path = 'D:\\data\\tensile_r4'

test_files = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt']

eps_arr = np.linspace(0.0, 0.011, 10001)
sig_arr = np.zeros_like(eps_arr)


for file_name in test_files:
    data = os.path.join(test_file_path, file_name)
    eps, sig_tex = np.loadtxt(data)
    sig_arr += np.interp(eps_arr, eps, sig_tex) / 6.


def get_ab(k):
    a = (0.003 * (170000. - k) - (624.225 - 0.006 * k)) / (0.003 * 0.006 ** 2)
    b = (170000. - k - 3. * a * 0.006 ** 2) / (2. * 0.006)
    return a, b, k


a, b, k = get_ab(40000.)


def reinf_law(x):
    b1 = a * x ** 3 + b * x ** 2 + k * x
    b2 = a * 0.006 ** 3 + b * \
        0.006 ** 2 + k * 0.006 + 170000. * (x - 0.006)
    return b1 * (x < 0.006) + b2 * (x >= 0.006)

print(a, b)

plt.plot(eps_arr, sig_arr)
plt.plot(eps_arr, reinf_law(eps_arr))
plt.ylim(0, 1500)

plt.show()
