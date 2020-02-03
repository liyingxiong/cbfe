'''
Created on 15.02.2017

@author: Yingxiong
'''

import matplotlib.pyplot as plt
import numpy as np
from cbfe.scratch.fe_nls_solver_incre1 import MATSEval, FETS1D52ULRH, TStepper, TLoop

from ibvpy.api import BCDof

ts = TStepper(n_e_x=200.)
n_dofs = ts.domain.n_dofs
tl = TLoop(ts=ts)


print(n_dofs)


def predict(L_x, u, slip, bond):

    tl.ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),
                     BCDof(var='u', dof=n_dofs - 1, value=u)]
    tl.ts.L_x = L_x
    tl.ts.mats_eval.slip = slip.tolist()
    tl.ts.mats_eval.bond = bond.tolist()
    U_record, F_record, sig_record = tl.eval()
    return U_record, F_record, sig_record

for bond2, alpha in zip([30., 50., 70., 90, 110], [0.2, 0.4, 0.6, 0.8, 1.0]):

    slip = np.array([0., 0.5, 1., 1.5, 2.0, 2.5, 3.0])
    bond = np.array([0., 25., 40., 53, 65, 75, bond2])

    plt.subplot(221)
    plt.plot(slip, bond, 'k', alpha=alpha)
    plt.xlabel('slip')
    plt.ylabel('bond')

    U1, F1, sig1 = predict(1500, 3., slip, bond)

    plt.subplot(222)
    plt.plot(U1[:, n_dofs - 1], F1[:, n_dofs - 1], 'k', lw=2, alpha=alpha)
    plt.xlabel('displacement')
    plt.ylabel('pull-out force')

    plt.subplot(223)
    X = np.linspace(0, ts.L_x, ts.n_e_x + 1)
    u1_node = np.reshape(U1[-1, :], (-1, 2)).T

    plt.plot(X,  u1_node[1] - u1_node[0], 'k', alpha=alpha)

    plt.xlabel('x')
    plt.ylabel('slip')

    plt.subplot(224)
    X_ip = np.repeat(X, 2)[1:-1]
    plt.plot(X_ip, sig1[-1, :],  'k', alpha=alpha)
    plt.xlabel('x')
    plt.ylabel('shear flow',)


plt.show()
