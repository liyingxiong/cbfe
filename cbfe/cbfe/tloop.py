'''
Created on 12.01.2016

@author: Yingxiong
'''
import numpy as np
from traits.api import implements, Int, Array, HasTraits, Instance, \
    Property, cached_property, Constant, Float, List
from .tstepper import TStepper


class TLoop(HasTraits):

    ts = Instance(TStepper)
    d_t = Float(0.01)
    t_max = Float(1.0)
    k_max = Int(50)
    tolerance = Float(1e-8)

    def eval(self):

        self.ts.apply_essential_bc()

        t_n = 0.
        t_n1 = t_n
        n_dofs = self.ts.domain.n_dofs
        n_e = self.ts.domain.n_active_elems
        n_ip = self.ts.fets_eval.n_gp
        n_s = self.ts.mats_eval.n_s
        U_k = np.zeros(n_dofs)
        eps = np.zeros((n_e, n_ip, n_s))
        sig = np.zeros((n_e, n_ip, n_s))
        alpha = np.zeros((n_e, n_ip))
        q = np.zeros((n_e, n_ip))
        kappa = np.zeros((n_e, n_ip))

        U_record = np.zeros(n_dofs)
        F_record = np.zeros(n_dofs)
        sf_record = np.zeros(2 * n_e)
        t_record = [t_n]
        eps_record = [np.zeros_like(eps)]
        sig_record = [np.zeros_like(sig)]

        while t_n1 <= self.t_max - self.d_t:
            print('==================')
            t_n1 = t_n + self.d_t
            k = 0
            scale = 1.0
            step_flag = 'predictor'
            d_U = np.zeros(n_dofs)
            d_U_k = np.zeros(n_dofs)
            while k <= self.k_max:
                print(k)
                # if k == self.k_max:  # handling non-convergence
                #                     scale *= 0.5
                # print scale
                #                     t_n1 = t_n + scale * self.d_t
                #                     k = 0
                #                     d_U = np.zeros(n_dofs)
                #                     d_U_k = np.zeros(n_dofs)
                #                     step_flag = 'predictor'
                #                     eps = eps_r
                #                     sig = sig_r
                #                     alpha = alpha_r
                #                     q = q_r
                #                     kappa = kappa_r

                R, K, eps, sig, alpha, q, kappa = self.ts.get_corr_pred(
                    step_flag, U_k, d_U_k, eps, sig, t_n, t_n1, alpha, q, kappa)

                F_ext = -R
                K.apply_constraints(R)
#                 print 'r', np.linalg.norm(R)
                d_U_k = K.solve()
                d_U += d_U_k
#                 print 'r', np.linalg.norm(R)
                if np.linalg.norm(R) < self.tolerance:
                    F_record = np.vstack((F_record, F_ext))
                    U_k += d_U
                    U_record = np.vstack((U_record, U_k))
                    sf_record = np.vstack((sf_record, sig[:, :, 1].flatten()))
                    eps_record.append(np.copy(eps))
                    sig_record.append(np.copy(sig))
                    t_record.append(t_n1)
                    break
                k += 1
                step_flag = 'corrector'

            t_n = t_n1
        return U_record, F_record, sf_record, np.array(t_record), eps_record, sig_record

if __name__ == '__main__':

    from matplotlib import pyplot as plt
    from ibvpy.api import BCDof

    ts = TStepper()

    n_dofs = ts.domain.n_dofs

#     tf = lambda t: 1 - np.abs(t - 1)
#     ts.bc_list = [BCDof(var='u', dof=0, value=0.0),
# BCDof(var='u', dof=n_dofs - 1, value=2.5, time_function=tf)]

    ts.bc_list = [BCDof(var='u', dof=0, value=0.0),
                  BCDof(var='f', dof=n_dofs - 1, value=0.5)]

    tl = TLoop(ts=ts, d_t=0.01)

    U_record, F_record, sf_record, t_record, eps_record, sig_record = tl.eval()
    n_dof = 2 * ts.domain.n_active_elems + 1
    print('U_record', U_record[:, n_dof])
    print('F_record', F_record[:, n_dof])
#     print U_record[:, n_dof]
#     print F_record[:, n_dof]
    plt.plot(U_record[:, n_dof], F_record[:, n_dof] / 1000., marker='.')
#     plt.ylim(0, 35)
    plt.xlabel('displacement')
    plt.ylabel('force')
    plt.show()
