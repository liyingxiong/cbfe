'''
Created on 09.12.2017

@author: Yingxiong
'''
from traits.api import implements, Int, Array, HasTraits, Instance, \
    Property, cached_property, Constant, Float, List
import numpy as np
from cbfe.fets1d52ulrh import FETS1D52ULRH
from ibvpy.api import BCDof
import matplotlib.pyplot as plt
from ibvpy.api import BCDof
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly
from ibvpy.mesh.fe_grid import FEGrid
from scipy.misc import derivative
from scipy.optimize import brentq, brent, fsolve
# import RuntimeWarning
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)


class MATSEval(HasTraits):
    E_m = Float(28484, tooltip='Stiffness of the matrix [MPa]',
                auto_set=False, enter_set=False)

    E_f = Float(170000, tooltip='Stiffness of the fiber [MPa]',
                auto_set=False, enter_set=False)

    E_b = Float(5,
                tooltip='Bond stiffness [N/mm]')

    sigma_y = Float(5,
                    label="sigma_y",
                    desc="Yield stress",
                    enter_set=True,
                    auto_set=False)

    def A(self, alpha):
        return np.interp(alpha, [-4.2, -1.6, 0, 1.6, 4.2],  [1., -2, 0, 2, -1])

    def get_corr_pred(self, eps, d_eps, sig, t_n, t_n1, alpha, q, kappa):
        #         g = lambda k: 0.8 - 0.8 * np.exp(-k)
        #         g = lambda k: 1. / (1 + np.exp(-2 * k + 6.))
        n_e, n_ip, n_s = eps.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[:, :, 0, 0] = self.E_m
        D[:, :, 2, 2] = self.E_f
        sig_trial = sig[:, :, 1] + self.E_b * d_eps[:,:, 1]
        f_trial = abs(sig_trial - self.A(alpha)) - (self.sigma_y)
        elas = f_trial <= 1e-8
        plas = f_trial > 1e-8
        d_sig = np.einsum('...st,...t->...s', D, d_eps)
        sig += d_sig

#         delta_f = lambda d_kappa: f_trial - self.E_b * \
#             d_kappa - self.A(alpha + d_kappa) + self.A(alpha)
#         try:
#             d_gamma = fsolve(delta_f, np.zeros_like(alpha))
#         except:
#             print f_trial.shape
#             print alpha.shape

        K = derivative(self.A, alpha, dx=1e-6)

        d_gamma = f_trial / (self.E_b + K) * plas

        sig[:, :, 1] = sig_trial - d_gamma * self.E_b * np.sign(sig_trial - self.A(alpha))
        alpha += d_gamma * np.sign(sig_trial - self.A(alpha))

        E_p = self.E_b * K / (self.E_b + K)
        D[:, :, 1, 1] = self.E_b * elas + E_p * plas

        return sig, D, alpha, q, kappa

    n_s = Constant(3)


class TStepper(HasTraits):

    mats_eval = Instance(MATSEval, arg=(), kw={})  # material model

    fets_eval = Instance(FETS1D52ULRH, arg=(), kw={})  # element formulation

    A = Property()
    '''array containing the A_m, L_b, A_f
    '''

    def _get_A(self):
        return np.array([self.fets_eval.A_m, self.fets_eval.P_b, self.fets_eval.A_f])

    # Number of elements
    n_e_x = 30
    # length
    L_x = Float(600.0)

    domain = Property(Instance(FEGrid), depends_on='L_x')
    '''Diescretization object.
    '''
    @cached_property
    def _get_domain(self):
        # Element definition
        domain = FEGrid(coord_max=(self.L_x,),
                        shape=(self.n_e_x,),
                        fets_eval=self.fets_eval)
        return domain

    bc_list = List(Instance(BCDof))

    J_mtx = Property(depends_on='L_x')
    '''Array of Jacobian matrices.
    '''
    @cached_property
    def _get_J_mtx(self):
        fets_eval = self.fets_eval
        domain = self.domain
        # [ d, n ]
        geo_r = fets_eval.geo_r.T
        # [ d, n, i ]
        dNr_geo = geo_r[:, :, None] * np.array([1, 1]) * 0.5
        # [ i, n, d ]
        dNr_geo = np.einsum('dni->ind', dNr_geo)
        # [ n_e, n_geo_r, n_dim_geo ]
        elem_x_map = domain.elem_X_map
        # [ n_e, n_ip, n_dim_geo, n_dim_geo ]
        J_mtx = np.einsum('ind,enf->eidf', dNr_geo, elem_x_map)
        return J_mtx

    J_det = Property(depends_on='L_x')
    '''Array of Jacobi determinants.
    '''
    @cached_property
    def _get_J_det(self):
        return np.linalg.det(self.J_mtx)

    B = Property(depends_on='L_x')
    '''The B matrix
    '''
    @cached_property
    def _get_B(self):
        '''Calculate and assemble the system stiffness matrix.
        '''
        mats_eval = self.mats_eval
        fets_eval = self.fets_eval
        domain = self.domain

        n_s = mats_eval.n_s

        n_dof_r = fets_eval.n_dof_r
        n_nodal_dofs = fets_eval.n_nodal_dofs

        n_ip = fets_eval.n_gp
        n_e = domain.n_active_elems
        #[ d, i]
        r_ip = fets_eval.ip_coords[:, :-2].T
        # [ d, n ]
        geo_r = fets_eval.geo_r.T

        J_inv = np.linalg.inv(self.J_mtx)

        # shape function for the unknowns
        # [ d, n, i]
        Nr = 0.5 * (1. + geo_r[:, :, None] * r_ip[None,:])
        dNr = 0.5 * geo_r[:, :, None] * np.array([1, 1])

        # [ i, n, d ]
        Nr = np.einsum('dni->ind', Nr)
        dNr = np.einsum('dni->ind', dNr)
        Nx = Nr
        # [ n_e, n_ip, n_dof_r, n_dim_dof ]
        dNx = np.einsum('eidf,inf->eind', J_inv, dNr)

        B = np.zeros((n_e, n_ip, n_dof_r, n_s, n_nodal_dofs), dtype='f')
        B_N_n_rows, B_N_n_cols, N_idx = [1, 1], [0, 1], [0, 0]
        B_dN_n_rows, B_dN_n_cols, dN_idx = [0, 2], [0, 1], [0, 0]
        B_factors = np.array([-1, 1], dtype='float_')
        B[:, :,:, B_N_n_rows, B_N_n_cols] = (B_factors[None, None,:] *
                                              Nx[:, :, N_idx])
        B[:, :,:, B_dN_n_rows, B_dN_n_cols] = dNx[:,:,:, dN_idx]
        return B

    def apply_essential_bc(self):
        '''Insert initial boundary conditions at the start up of the calculation.. 
        '''
        self.K = SysMtxAssembly()
        for bc in self.bc_list:
            bc.apply_essential(self.K)

    def apply_bc(self, step_flag, K_mtx, F_ext, t_n, t_n1):
        '''Apply boundary conditions for the current load increement
        '''
        for bc in self.bc_list:
            bc.apply(step_flag, None, K_mtx, F_ext, t_n, t_n1)

    def get_corr_pred(self, step_flag, U, d_U, eps, sig, t_n, t_n1, alpha, q, kappa):
        '''Function calculationg the residuum and tangent operator.
        '''
        mats_eval = self.mats_eval
        fets_eval = self.fets_eval
        domain = self.domain
        elem_dof_map = domain.elem_dof_map

        n_e = domain.n_active_elems
        n_dof_r, n_dim_dof = self.fets_eval.dof_r.shape
        n_nodal_dofs = self.fets_eval.n_nodal_dofs
        n_el_dofs = n_dof_r * n_nodal_dofs
        # [ i ]
        w_ip = fets_eval.ip_weights

        d_u_e = d_U[elem_dof_map]
        #[n_e, n_dof_r, n_dim_dof]
        d_u_n = d_u_e.reshape(n_e, n_dof_r, n_nodal_dofs)
        #[n_e, n_ip, n_s]
        d_eps = np.einsum('einsd,end->eis', self.B, d_u_n)

        # update strain ---this should be integrated into the material model
        eps += d_eps

        # material response state variables at integration point
        sig, D, alpha, q, kappa = mats_eval.get_corr_pred(
            eps, d_eps, sig, t_n, t_n1, alpha, q, kappa)

        # system matrix
        self.K.reset_mtx()
        Ke = np.einsum('i,s,einsd,eist,eimtf,ei->endmf',
                       w_ip, self.A, self.B, D, self.B, self.J_det)

        self.K.add_mtx_array(
            Ke.reshape(-1, n_el_dofs, n_el_dofs), elem_dof_map)

        # internal forces
        # [n_e, n_n, n_dim_dof]
        Fe_int = np.einsum('i,s,eis,einsd,ei->end',
                           w_ip, self.A, sig, self.B, self.J_det)
        F_int = -np.bincount(elem_dof_map.flatten(), weights=Fe_int.flatten())
        self.apply_bc(step_flag, self.K, F_int, t_n, t_n1)
        return F_int, self.K, eps, sig, alpha, q, kappa


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
            t_n1 = t_n + self.d_t
            k = 0
            scale = 1.0
            step_flag = 'predictor'
            d_U = np.zeros(n_dofs)
            d_U_k = np.zeros(n_dofs)
            while k <= self.k_max:
                if k == self.k_max:  # handling non-convergence
                    print(t_n1)
                    print('non-convergence')
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
        return U_record, F_record, sf_record, np.array(t_record), np.array(eps_record), np.array(sig_record)


if __name__ == '__main__':

    mat = MATSEval()

    fet = FETS1D52ULRH(A_m=120. * 13. - 9. * 1.85,
                       P_b=10.,
                       A_f=9. * 1.85)

    ts = TStepper(mats_eval=mat,
                  fets_eval=fet)

    ts.L_x = 400.
    ts.n_e_x = 20

    n_dofs = ts.domain.n_dofs

#     d_array = np.array([0., 2, 3, 4, 5])

    d_array = np.array(
        [0., 2, 0.252, 4., 1.62, 5])
    dd_arr = np.abs(np.diff(d_array))
    x = np.hstack((0, np.cumsum(dd_arr) / sum(dd_arr)))
    from scipy.interpolate import interp1d
    tf = interp1d(x, d_array)

    ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),
                  BCDof(var='u', dof=n_dofs - 1, value=1., time_function=tf)]

    tl = TLoop(ts=ts, d_t=0.002)

    U_record, F_record, sf_record, t_record, eps_record, sig_record = tl.eval()
    n_dof = 2 * ts.domain.n_active_elems + 1
    plt.plot(U_record[:, n_dof], F_record[
             :, n_dof] / 1000., label='loaded end')
#     plt.plot(U_record[:, 1], F_record[:, n_dof] / 1000., label='free end')

    plt.xlabel('displacement [mm]')
    plt.ylabel('pull-out force [KN]')

    plt.legend(loc='best')

    plt.figure()
    plt.plot(eps_record[:, -1, -1, 1], sig_record[:, -1, -1,  1])

    plt.show()
