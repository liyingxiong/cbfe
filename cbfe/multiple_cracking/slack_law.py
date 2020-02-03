'''
Created on 04.04.2017

@author: Yingxiong
'''
import os
import numpy as np
import matplotlib.pyplot as plt

test_file_path = 'D:\\data\\tensile_r4'

test_files = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt']

eps_arr = np.linspace(0.0, 0.01, 10001)
sig_arr = np.zeros_like(eps_arr)


for file_name in test_files:
    data = os.path.join(test_file_path, file_name)
    eps, sig_tex = np.loadtxt(data)
    sig_arr += np.interp(eps_arr, eps, sig_tex) / 6.

t = np.diff(sig_arr) / np.diff(eps_arr)

print(np.interp(0.006, eps_arr, sig_arr))


t_m = t[241]
t_line_x = np.linspace(-0.005, 0.005, 1000)
t_line_y = t[241] * t_line_x

t_line_x += eps_arr[242]
t_line_y += sig_arr[242]

t_line_0 = sig_arr[1::] - t * eps_arr[1::]

a = np.hstack((t, t[-1])) / (2. * eps_arr)
a_y = a * eps_arr ** 2
# print a[1725]
# print np.argmin(np.abs(a_y[100::] - sig_arr[100::]))

plt.plot(eps_arr, sig_arr, 'b')
plt.plot(eps_arr, 49271503.9482 * eps_arr ** 2)
plt.plot(eps_arr, a_y)
# plt.plot(t_line_x, t_line_y, 'b')

# plt.plot(eps_arr, np.hstack((0, t_line_0)))
# print t[241]
# print np.where(np.abs(t_line_0) < 1)
# plt.xlim(0,)
# plt.ylim(0,)

law1 = 10231172.3766 * eps_arr[0:242]
law2 = sig_arr[242::]
law = np.hstack((law1, law2))

# print law
# print eps_arr
#
# law_x = np.linspace(0, 0.01, 25)
# law_y = np.interp(law_x, eps_arr, law)

# print [law_x]
# print [law_y]

# plt.plot(eps_arr, law)
# plt.plot(law_x, law_y)

plt.show()
