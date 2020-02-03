'''
Created on 13.10.2017

@author: Yingxiong
'''
import io
import numpy as np
import matplotlib.pyplot as plt
from inverse.fem_inverse import TStepper, TLoop, MATSEval, FETS1D52ULRH
from ibvpy.api import BCDof


def calib(w_arr, pf_arr, l):

    mat = MATSEval(E_m=2000.,  # MPa
                   E_f=71000.)  # MPa

    fet = FETS1D52ULRH(A_m=20000.,  # mm2
                       A_f=6.91)  # mm2

    ts = TStepper(mats_eval=mat,
                  fets_eval=fet,
                  L_x=l,  # speciment length
                  n_e_x=20  # number of elements
                  )

    n_dofs = ts.domain.n_dofs

    ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),  # the fixed DOF
                  BCDof(var='u', dof=n_dofs - 1, value=1.0)]  # the DOF on which the displacement is applied

    tl = TLoop(ts=ts, w_arr=w_arr, pf_arr=pf_arr)

    slip, bond = tl.eval()

    return bond, slip


s1 = open("D:\\download\\curve1.txt").read().replace(',', '.')
data1 = np.loadtxt(io.StringIO(s1)).T

plt.plot(data1[0], data1[1], '--', label='original')

disp1 = np.hstack((0, data1[0][158:342]))
force1 = np.hstack((0, data1[1][158:342]))  # kN

plt.plot(disp1, force1,  lw=2, label='modified')

plt.xlabel('disp[mm]')
plt.ylabel('force[kN]')
plt.legend(loc='best')

plt.show()

w_arr501 = np.linspace(0., 0.1, 31)
pf_arr501 = np.interp(w_arr501, disp1,  force1) * 1000.  # N

bond501,  slip501 = calib(w_arr501, pf_arr501, 50.)

plt.plot(bond501, slip501)

# s2 = open("D:\\download\\curve2.txt").read().replace(',', '.')
# data2 = np.loadtxt(StringIO.StringIO(s2)).T
#
#
# plt.plot(data2[0], data2[1])
# plt.show()
#
#
# plt.plot(np.arange(len(data2[1])), data2[1])
# plt.show()
#
# disp2 = np.hstack((0, data2[0][200:315]))
# force2 = np.hstack((0, data2[1][200:315]))  # kN

# plt.plot(disp2, force2)
# plt.show()


# w_arr502 = np.linspace(0., 0.1, 21)
# pf_arr502 = np.interp(w_arr502, disp2,  force2) * 1000.  # N
#
# plt.plot(w_arr502, pf_arr502)
# plt.show()
#
#
# bond502,  slip502 = calib(w_arr502, pf_arr502, 50.)
#
#
# plt.plot(bond502, slip502)


plt.xlabel('slip [mm]')
plt.ylabel('bond [N/mm]')
plt.show()
