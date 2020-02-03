'''
Created on 01.04.2017

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt
import os
from .tensile_test import CompositeTensileTest
from .cb import NonLinearCB
from .fe_nls_solver_cb import MATSEval, FETS1D52ULRH, TStepper, TLoop
from stats.misc.random_field.random_field_1D import RandomField
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == '__main__':
    fets = FETS1D52ULRH(A_m=9. * 95. - 18 * 1.84,
                        A_f=18 * 1.84,
                        L_b=1)
    ts = TStepper(fets_eval=fets)

    bond_9 = [0.,  31.77189817,  42.50645201,  48.25769046,
              50.14277087,  49.43560803,  48.67833003,  46.19192825,
              43.9328575,  42.11038798,  41.89493604,  41.78074725,
              42.12809587,  42.4754445,  42.68052187,  42.86052074,
              42.97271023,  43.08489972,  43.1013979,  43.10971771,  43.0578219]

    for alpha in [1., 2., 4., ]:

        bond_18 = [alpha * bond for bond in bond_9]

        cb = NonLinearCB(A_c=9. * 95.,
                         tstepper=ts,
                         max_w_p=1.84 * 18 * 1500,  # [N]
                         slip=[0.,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,
                               1., 1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2.],
                         bond=bond_18,
                         n_BC=100)

        random_field = RandomField(seed=False,
                                   lacor=1.,
                                   length=2000,
                                   nx=1000,
                                   nsim=1,
                                   loc=.0,
                                   stdev=0.05,
                                   mean=5.,
                                   distr_type='Gauss')

        ctt = CompositeTensileTest(n_x=1000,
                                   L=2000,
                                   cb=cb,
                                   sig_mu_x=random_field.random_field,
                                   strength=1.84 * 18. * 1500. / 9. / 95.)

        sig_c_i, z_x_i, BC_x_i, sig_c_u, n_crack = ctt.get_cracking_history()
        load_arr1 = np.unique(
            np.hstack((np.linspace(0, sig_c_u, 100), sig_c_i)))
        eps_c_arr1 = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr1)
        plt.plot(eps_c_arr1, load_arr1)

        w = ctt.get_w_dist(sig_c_i, z_x_i, BC_x_i, [sig_c_u])

        print([np.interp(sig_c_i, load_arr1, eps_c_arr1)])

        print(w)

#     random_field = RandomField(seed=False,
#                                lacor=1.,
#                                length=2000,
#                                nx=1000,
#                                nsim=1,
#                                loc=.0,
#                                stdev=0.05,
#                                mean=5.,
#                                distr_type='Gauss')
#
#     ctt.sig_mu_x = random_field.random_field
#
#     sig_c_i, z_x_i, BC_x_i, sig_c_u, n_crack = ctt.get_cracking_history()
#     load_arr2 = np.unique(np.hstack((np.linspace(0, sig_c_u, 100), sig_c_i)))
#     eps_c_arr2 = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr2)
#     plt.plot(eps_c_arr2, load_arr2)
#
#     w = ctt.get_w_dist(sig_c_i, z_x_i, BC_x_i, [sig_c_u])
#
#     print w
#
#     random_field = RandomField(seed=False,
#                                lacor=1.,
#                                length=2000,
#                                nx=1000,
#                                nsim=1,
#                                loc=.0,
#                                stdev=0.05,
#                                mean=5.,
#                                distr_type='Gauss')
#
#     ctt.sig_mu_x = random_field.random_field
#
#     sig_c_i, z_x_i, BC_x_i, sig_c_u, n_crack = ctt.get_cracking_history()
#     load_arr3 = np.unique(np.hstack((np.linspace(0, sig_c_u, 100), sig_c_i)))
#     eps_c_arr3 = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr3)
#     plt.plot(eps_c_arr3, load_arr3)
#
#     w = ctt.get_w_dist(sig_c_i, z_x_i, BC_x_i, [sig_c_u])
#
#     print w

    plt.ylim(0, 60)
    plt.show()
