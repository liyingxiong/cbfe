'''
Created on 19.03.2017

@author: Yingxiong
'''
from traits.api import implements, Int, Array, HasTraits, Instance, \
    Property, cached_property, Constant, Float, List
from .cb import NonLinearCB
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize_scalar, fmin, brute, newton
import matplotlib.pyplot as plt
from stats.misc.random_field.random_field_1D import RandomField


class CompositeTensileTest(HasTraits):

    cb = Instance(NonLinearCB)  # crack bridge model

    #=========================================================================
    # Discretization of the specimen
    #=========================================================================
    n_x = Int(501)  # number of material points
    L = Float(1000.)  # the specimen length - mm
    x = Property(depends_on='n_x, L')  # coordinates of the material points

    cs = Float(18.)  # predefined crack spacing

    @cached_property
    def _get_x(self):
        return np.linspace(0, self.L, self.n_x)
    sig_mu_x = Array()

    #=========================================================================
    # Description of cracked specimen
    #=========================================================================
    y = List([])  # the list to record the crack positions
    z_x = Property(depends_on='n_x, L, y')
    '''the array containing the distances from each material point to its 
    nearest crack position
    '''
    @cached_property
    def _get_z_x(self):
        try:
            y = np.array(self.y)
            z_grid = np.abs(self.x[:, np.newaxis] - y[np.newaxis, :])
            return np.amin(z_grid, axis=1)
        except ValueError:  # no cracks exist
            return np.ones_like(self.x) * 2 * self.L

    BC_x = Property(depends_on='x, y, L')
    '''the array containing the boundary condition for each material point
    '''
    @cached_property
    def _get_BC_x(self):
        try:
            y = np.sort(self.y)
            d = (y[1:] - y[:-1]) / 2.0
            # construct a piecewise function for interpolation
            xp = np.hstack([self.x[0], y, self.x[-1]])
            BC_arr = np.hstack([y[0], d, self.L - y[-1], np.NAN])
            f = interp1d(xp, BC_arr, kind='zero')
            return f(self.x)
        except IndexError:
            return np.vstack([np.zeros_like(self.x), np.zeros_like(self.x)])

    #=========================================================================
    # Strength of the specimen
    #=========================================================================
    strength = Float(20.)  # [MPa]

    #=========================================================================
    # Determine the cracking load level
    #=========================================================================
    def get_sig_c_z(self, sig_mu, z, BC, sig_c_i_1):
        '''Determine the composite remote stress initiating a crack 
        at position z'''
        fun = lambda sig_c: sig_mu - self.cb.get_sig_m_z(z, BC, sig_c)
        try:
            # search the cracking stress level between zero and ultimate
            # composite stress
            return brentq(fun, 0, self.strength)

        except:
            # no solution, shielded zone
            return 1e6

    get_sig_c_x_i = np.vectorize(get_sig_c_z)

    def get_sig_c_i(self, sig_c_i_1):
        '''Determine the new crack position and level of composite stress
        '''
        # for each material point find the load factor initiating a crack
        sig_c_x_i = self.get_sig_c_x_i(self, self.sig_mu_x,
                                       self.z_x, self.BC_x, sig_c_i_1)
        # get the position of the material point corresponding to
        # the minimum cracking load factor
        y_idx = np.argmin(sig_c_x_i)
        y_i = self.x[y_idx]
        sig_c_i = sig_c_x_i[y_idx]
        return sig_c_i, y_i

    #=========================================================================
    # determine the crack history
    #=========================================================================
    def get_cracking_history(self):
        '''Trace the response crack by crack.
        '''
        z_x_lst = [self.z_x]  # record z array of each cracking state
        # record boundary condition of each cracking state
        BC_x_lst = [self.BC_x]
        sig_c_lst = [0.]  # record cracking load factor

        # min matrix strength
        np.amin(self.sig_mu_x)
        sig_c_0 = np.amin(self.sig_mu_x) * self.cb.E_c / self.cb.E_m

        # introduce the predefined cracks
        cracks = self.cs * np.arange(self.L / self.cs + 1)
        d = np.abs(self.x[:, None] - cracks[None, :])
        min_idx = np.argmin(d, axis=0)
        crack_postion = self.x[min_idx]

        for i, crack in enumerate(crack_postion):
            self.y.append(crack)
            # add a small number to the min matrix strength to avoid numerical
            # problems
            sig_c_lst.append(sig_c_0 + i * 1e-8)
            z_x_lst.append(np.array(self.z_x))
            BC_x_lst.append(np.array(self.BC_x))

        # determine the following cracking load factors
        while True:
            sig_c_i, y_i = self.get_sig_c_i(sig_c_lst[-1])
            if sig_c_i >= self.strength or sig_c_i == 1e6:
                break
            print(sig_c_i, y_i)
            self.y.append(y_i)
            print('number of cracks:', len(self.y))
            sig_c_lst.append(sig_c_i)
            z_x_lst.append(np.array(self.z_x))
            BC_x_lst.append(np.array(self.BC_x))
#             self.save_cracking_history(sig_c_i, z_x_lst, BC_x_lst)
#             print 'strength', self.strength
        print('cracking history determined')
        sig_c_u = self.strength
        print(sig_c_u)
        n_cracks = len(self.y)
        self.y = []
        return np.array(sig_c_lst), np.array(z_x_lst), BC_x_lst, sig_c_u, n_cracks

    #=========================================================================
    # post processing
    #=========================================================================
    def get_eps_c_i(self, sig_c_i, z_x_i, BC_x_i):
        '''For each cracking level calculate the corresponding
        composite strain eps_c.
        '''
        return np.array([np.trapz(self.get_eps_f_x(sig_c, z_x, BC_x), self.x) / self.L
                         for sig_c, z_x, BC_x in zip(sig_c_i, z_x_i, BC_x_i)
                         ])

    def get_eps_f_x(self, sig_c, z_x, BC):
        '''function to evaluate specimen reinforcement strain profile
        at given load level and crack distribution
        '''
        eps_f = np.zeros_like(self.x)
        z_x_map = np.argsort(z_x)
        eps_f[z_x_map] = self.cb.get_eps_f_z(
            z_x[z_x_map], BC, sig_c)
        return eps_f

    def get_sig_m_x(self, sig_c, z_x, BC):
        '''function to evaluate specimen matrix stress profile
        at given load level and crack distribution
        '''
        eps_m = np.zeros_like(self.x)
        z_x_map = np.argsort(z_x)
        eps_m[z_x_map] = self.cb.get_sig_m_z(z_x[z_x_map], BC[z_x_map], sig_c)
        return eps_m

    def get_eps_c_arr(self, sig_c_i, z_x_i, BC_x_i, load_arr):
        '''function to evaluate the average specimen strain array corresponding 
        to the given load_arr
        '''
        eps_arr = np.ones_like(load_arr)
        for i, load in enumerate(load_arr):
            idx = np.searchsorted(sig_c_i, load) - 1
            z_x = z_x_i[idx]
            if np.any(z_x == 2 * self.L):  # no cracks
                eps_arr[i] = load / self.cb.E_c
                sig_m = self.cb.E_m * eps_arr[i] * np.ones_like(self.x)
            else:
                BC_x = BC_x_i[idx]
                eps_arr[i] = np.trapz(
                    self.get_eps_f_x(load, z_x, BC_x), self.x) / self.L
                sig_m = self.get_sig_m_x(load, z_x, BC_x)
            # save the cracking history
            save = True
            if save:
                plt.plot(self.x, sig_m)
                plt.plot(self.x, self.sig_mu_x)
                plt.ylim((0., 1.2 * np.max(self.sig_mu_x)))
                savepath = 'D:\cracking history\\1\\load_step' + \
                    str(i + 1) + '.png'
                plt.savefig(savepath)
                plt.clf()

        return eps_arr

if __name__ == '__main__':

    cb = NonLinearCB(
        slip=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
        bond=[0., 10., 20., 30., 40., 50.],
        n_BC=100)

    random_field = RandomField(seed=False,
                               lacor=1.,
                               length=500,
                               nx=1000,
                               nsim=1,
                               loc=.0,
                               shape=60.,
                               scale=1.5,
                               distr_type='Weibull')

    ctt = CompositeTensileTest(n_x=1000,
                               L=500,
                               cb=cb,
                               sig_mu_x=random_field.random_field)

    sig_c_i, z_x_i, BC_x_i, sig_c_u, n_crack = ctt.get_cracking_history()
    load_arr = np.unique(np.hstack((np.linspace(0, sig_c_u, 100), sig_c_i)))
    eps_c_arr = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
    plt.plot(eps_c_arr, load_arr, 'k', lw=2, label='v_f=1.5%')
    plt.show()
