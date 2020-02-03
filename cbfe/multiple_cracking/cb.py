'''
Created on 14.03.2017

@author: Yingxiong
'''
from ibvpy.api import BCDof
from scipy.interpolate import interp2d
from traits.api import Int, HasTraits, Instance, \
    Property, cached_property, Float, List

import numpy as np

from .fe_nls_solver_cb import MATSEval, FETS1D52ULRH, TStepper, TLoop


class NonLinearCB(HasTraits):

    tstepper = Instance(TStepper, arg=(), kw={})

    tloop = Property(Instance(TLoop), depends_on='tstepper')

    @cached_property
    def _get_tloop(self):
        ts = self.tstepper
        ts.mats_eval.slip = self.slip
        ts.mats_eval.bond = self.bond

        return TLoop(ts=self.tstepper)

    A_c = Float(120 * 13)  # the cross-sectional area [mm^2]

    E_m = Property(depends_on='tstepper')  # matrix modulus

    @cached_property
    def _get_E_m(self):
        return self.tstepper.mats_eval.E_m

    E_f = Property(depends_on='tstepper')  # reinforcement modulus

    @cached_property
    def _get_E_f(self):
        return self.tstepper.mats_eval.E_f

    E_c = Property(depends_on='tstepper')  # composite modulus

    @cached_property
    def _get_E_c(self):
        return self.E_m * self.tstepper.fets_eval.A_m / self.A_c + self.E_f * self.tstepper.fets_eval.A_f / self.A_c

    slip = List
    bond = List

    n_BC = Int(50)  # number of CBs

    L_max = Float(500)  # Maximum cb length

    L_min = Float(1.)  # minimum cb length

    BC_list = Property(depends_on='n_BC, L_max, L_min')

    @cached_property
    def _get_BC_list(self):
        return np.logspace(np.log10(self.L_min), np.log10(self.L_max), self.n_BC)

    # force [N] to control the maximum pull-out force
    max_w_p = Float(20 * 120 * 13)

    interps = Property(depends_on='tstepper')

    @cached_property
    def _get_interps(self):

        interps_m = []
        interps_f = []

        print('preparing interploaters...')

        for L in self.BC_list:

            w = self.max_w_p / \
                (self.tstepper.mats_eval.E_f * self.tstepper.fets_eval.A_f) * L \
                + self.tstepper.mats_eval.slack * L

            # Number of degrees of freedom
            n_dofs = self.tstepper.domain.n_dofs
            self.tstepper.bc_list = [BCDof(var='u', dof=0, value=0.0),
                                     BCDof(var='u', dof=1, value=0.0),
                                     BCDof(var='u', dof=n_dofs - 1, value=w)]
            self.tstepper.L_x = L
            U_record, F_record, tau_record, sig_m, eps_f = self.tloop.eval()
            sig_m = self.avg_sig(sig_m)
            eps_f = self.avg_eps(eps_f)

            X = np.linspace(0, L, self.tstepper.n_e_x + 1)

            sig_c = F_record[:, -1] / self.A_c

            interp_m = interp2d(
                X[::-1], sig_c, sig_m)

            interps_m.append(interp_m)

            interp_f = interp2d(
                X[::-1], sig_c, eps_f)

            interps_f.append(interp_f)

        print('complete')

        return [interps_m, interps_f]

    interps_m = Property(depends_on='tstepper')

    @cached_property
    def _get_interps_m(self):
        return self.interps[0]

    interps_f = Property(depends_on='tstepper')

    @cached_property
    def _get_interps_f(self):
        return self.interps[1]

    def avg_sig(self, sig):
        '''average the stress on the integration points to the nodes'''
        sig = np.hstack((sig[:, 0:1], sig, sig[:, -1::]))
        sig = (sig[:, 0::2] + sig[:, 1::2]) / 2.
        sig[:, -1] = 0.
        return sig

    def avg_eps(self, eps):
        '''average the strain on the integration points to the nodes'''
        eps = np.hstack((eps[:, 0:1], eps, eps[:, -1::]))
        eps = (eps[:, 0::2] + eps[:, 1::2]) / 2.
        return eps

    def get_interp_idx(self, l):
        i = min(np.sum(self.BC_list - l < 0), self.n_BC - 1)
        return i

    def get_sig_m_z(self, z_arr, BC, load):
        def get_sig_m_i(self, z, BC, load):
            idx = self.get_interp_idx(BC)
            interp = self.interps_m[idx]
            return interp(z, load)
        v = np.vectorize(get_sig_m_i)
        return v(self, z_arr, BC, load)

    def get_eps_f_z(self, z_arr, BC, load):
        def get_eps_f_i(self, z, BC, load):
            idx = self.get_interp_idx(BC)
            interp = self.interps_f[idx]
            return interp(z, load)
        v = np.vectorize(get_eps_f_i)
        return v(self, z_arr, BC, load)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp2d

    def test_error(test_l=5, sig_c=12):

        cb = NonLinearCB(
            slip=[0.,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,
                  1., 1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2.],
            #             bond=[0.,  31.77189817,  42.50645201,  48.25769046,
            #                   50.14277087,  49.43560803,  48.67833003,  46.19192825,
            #                   43.9328575,  42.11038798,  41.89493604,  41.78074725,
            #                   42.12809587,  42.4754445,  42.68052187,  42.86052074,
            # 42.97271023,  43.08489972,  43.1013979,  43.10971771,
            # 43.0578219],
            bond=[0.,  31.77189817,  42.50645201,  48.25769046,
                  50.14277087,  25.43560803,  10.67833003,  5.19192825,
                  3.9328575,  1.11038798,  1.89493604,  1.78074725,
                  1.12809587,  1.4754445,  1.68052187,  1.86052074,
                  1.97271023,  1.08489972,  1.1013979,  1.10971771,  1.0578219],
            n_BC=2)

        x = np.linspace(0, test_l, 30)

        interp = cb.interps_m[cb.get_interp_idx(test_l)]
        y = interp(x, sig_c)

        plt.plot(x, y)
        plt.legend()
        plt.xlabel('z')
        plt.ylabel('matrix stress')

        #============================================================

        cb.tstepper.L_x = test_l
        n_dofs = cb.tstepper.domain.n_dofs
        cb.tstepper.bc_list = [BCDof(var='u', dof=0, value=0.0),
                               BCDof(var='u', dof=1, value=0.0),
                               BCDof(var='u', dof=n_dofs - 1, value=0.5 + test_l * 0.005)]

        U_record, F_record, tau_record, sig_m, eps_f = cb.tloop.eval()

        X = np.linspace(0, test_l, cb.tstepper.n_e_x + 1)

        plt.plot(X[::-1], cb.avg_sig(sig_m)[-1, :])

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        x, y = np.meshgrid(X[::-1], F_record[:, -1] / 120. / 13.)

        surf = ax.plot_surface(
            x, y, cb.avg_eps(eps_f), rstride=10, cstride=1, cmap=cm.coolwarm)

        plt.show()

    test_error(400)
    plt.show()
