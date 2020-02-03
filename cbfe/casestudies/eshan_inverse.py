'''
Created on 25.12.2017

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt
from inverse.fem_inverse import MATSEval, FETS1D52ULRH, TStepper, TLoop
from ibvpy.api import BCDof
from scipy.interpolate import interp1d


mat = MATSEval(E_m=30000.,
               E_f=200000.)

fet = FETS1D52ULRH(A_m=250. * 250. - np.pi * 12.5 ** 2,
                   A_f=np.pi * 12.5 ** 2,
                   L_b=np.pi * 25.)

ts = TStepper(mats_eval=mat,
              fets_eval=fet,
              L_x=125.,  # half of speciment length
              n_e_x=20  # number of elements
              )

n_dofs = ts.domain.n_dofs

ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),  # the fixed DOF
              BCDof(var='u', dof=n_dofs - 1, value=1.0)]  # the DOF on which the displacement is applied


data = np.loadtxt('D:\\F-s-test.txt').T

x = data[1]
y = data[0]

x[0] = 0.
y[0] = 0.

interp = interp1d(x, y)

w_arr = np.hstack((0, 0.02, np.linspace(0.03, 1.15, 199)))

# w_arr = np.hstack((0, np.linspace(0.15, 1.15, 6)))

pf_arr = interp(w_arr) * 1000.
pf_arr[0] = 0.

tl = TLoop(ts=ts, w_arr=w_arr, pf_arr=pf_arr, regularization=True)

slip, bond = tl.eval()

np.set_printoptions(precision=4)
print('slip')
print([np.array(slip)])
print('bond')
print([np.array(bond)])
plt.plot(x, y * 1000. / np.pi / 25. / 125.)
plt.plot(slip, bond)
plt.xlabel('slip [mm]')
plt.ylabel('bond [N/mm]')
plt.figure()
plt.plot(x, y * 1000.)
plt.plot(w_arr, pf_arr)
plt.show()
