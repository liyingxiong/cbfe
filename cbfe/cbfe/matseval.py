'''
Created on 12.01.2016

@author: Yingxiong
'''

from traits.api import implements, Int, Array, HasTraits, Instance, \
    Property, cached_property, Constant, Float, List
import numpy as np
from scipy.misc import derivative


class MATSEval(HasTraits):

    E_m = Float(28484, tooltip='Stiffness of the matrix [MPa]',
                auto_set=False, enter_set=False)

    E_f = Float(170000, tooltip='Stiffness of the fiber [MPa]',
                auto_set=False, enter_set=False)

    E_b = Float(2.0, tooltip='Bond stiffness [MPa]')

    sigma_y = Float(1.05,
                    label="sigma_y",
                    desc="Yield stress",
                    enter_set=True,
                    auto_set=False)

    K_bar = Float(0.08,  # 191e-6,
                  label="K",
                  desc="Plasticity modulus",
                  enter_set=True,
                  auto_set=False)

    H_bar = Float(0.0,  # 191e-6,
                  label="H",
                  desc="Hardening modulus",
                  enter_set=True,
                  auto_set=False)
    # bond damage law
    alpha = Float(1.0)
    beta = Float(1.0)
    g = lambda self, k: 1. / (1 + np.exp(-self.alpha * k + 6.)) * self.beta
#     g = lambda self, k: 0.

    def get_corr_pred(self, eps, d_eps, sig, t_n, t_n1, alpha, q, kappa):
        #         g = lambda k: 0.8 - 0.8 * np.exp(-k)
        #         g = lambda k: 1. / (1 + np.exp(-2 * k + 6.))

        print('matseval called')
        n_e, n_ip, n_s = eps.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[:, :, 0, 0] = self.E_m
        D[:, :, 2, 2] = self.E_f
        sig_trial = sig[:, :, 1]/(1-self.g(kappa)) + self.E_b * d_eps[:,:, 1]
        xi_trial = sig_trial - q
        f_trial = abs(xi_trial) - (self.sigma_y + self.K_bar * alpha)
        elas = f_trial <= 1e-8
        plas = f_trial > 1e-8
        d_sig = np.einsum('...st,...t->...s', D, d_eps)
        sig += d_sig

        d_gamma = f_trial / (self.E_b + self.K_bar + self.H_bar) * plas
        alpha += d_gamma
        kappa += d_gamma
        q += d_gamma * self.H_bar * np.sign(xi_trial)
        w = self.g(kappa)

        sig_e = sig_trial - d_gamma * self.E_b * np.sign(xi_trial)
        sig[:, :, 1] = (1-w)*sig_e

        E_p = -self.E_b / (self.E_b + self.K_bar + self.H_bar) * derivative(self.g, kappa, dx=1e-6) * sig_e \
            + (1 - w) * self.E_b * (self.K_bar + self.H_bar) / \
            (self.E_b + self.K_bar + self.H_bar)

        D[:, :, 1, 1] = (1-w)*self.E_b*elas + E_p*plas

        return sig, D, alpha, q, kappa

    def get_bond_slip(self):
        '''for plotting the bond slip relationship
        '''
#         s_arr = np.hstack((np.linspace(0, 10, 200),
# np.linspace(10., 10. - self.sigma_y / self.E_b, 10)))

        s_arr = np.hstack((np.linspace(0, 5, 200),
                           np.linspace(5., -5, 400)))

        sig_e_arr = np.zeros_like(s_arr)
        sig_n_arr = np.zeros_like(s_arr)
        w_arr = np.zeros_like(s_arr)

        sig_e = 0.
        alpha = 0.
        kappa = 0.

        for i in range(1, len(s_arr)):
            d_eps = s_arr[i] - s_arr[i - 1]
            sig_e_trial = sig_e + self.E_b * d_eps
            f_trial = abs(sig_e_trial) - (self.sigma_y + self.K_bar * alpha)
            if f_trial <= 1e-8:
                sig_e = sig_e_trial
            else:
                d_gamma = f_trial / (self.E_b + self.K_bar)
                alpha += d_gamma
                kappa += d_gamma
                sig_e = sig_e_trial - d_gamma * self.E_b * np.sign(sig_e_trial)
            w = self.g(kappa)
            w_arr[i] = w
            sig_n_arr[i] = (1. - w) * sig_e
            sig_e_arr[i] = sig_e

        return s_arr, sig_n_arr, sig_e_arr, w_arr

    n_s = Constant(3)

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    mat = MATSEval()

    slip, sig_n_arr, sig_e_arr, w_arr = mat.get_bond_slip()
    plt.plot(slip, sig_n_arr)
    plt.plot(slip, sig_e_arr, '--')
    plt.plot(slip, w_arr, '--')
    plt.title('bond-slip law')
    plt.show()
