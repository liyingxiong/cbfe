from envisage.ui.workbench.api import WorkbenchApplication
from mayavi.sources.api import VTKDataSource, VTKFileReader

from traits.api import implements, Int, Array, HasTraits, Instance, \
    Property, cached_property, Constant, Float, List

from ibvpy.api import BCDof
from ibvpy.fets.fets_eval import FETSEval, IFETSEval
from ibvpy.mats.mats1D import MATS1DElastic
from ibvpy.mats.mats1D5.mats1D5_bond import MATS1D5Bond
from ibvpy.mesh.fe_grid import FEGrid
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly
import matplotlib.pyplot as plt
import numpy as np
import sys
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
#     g = lambda self, k: 0.8 - 0.8 * np.exp(-k)
#     g = lambda self, k: 1. / (1 + np.exp(-0.7 * k + 6.))
    g = lambda self, k: 1. / \
        (1 + np.exp(-0.9 * k + 6.)) * 0.5 + \
        0.3 * (k > 0.1) + 2 * k * (k <= 0.1)

    # reiforcement damage law
    def g_f(self, eps_f, eps_f_max):
        e = abs(eps_f) * (abs(eps_f) > eps_f_max) + \
            eps_f_max * (abs(eps_f) <= eps_f_max)
        w_f = 50. * (e - 0.0015) * (e > 0.0015) * \
            (e <= 0.0215) + 1. * (e > 0.0215)
        return w_f, e
#         return 0, e

    # derivative of g_f
    def d_g_f(self, eps_f, eps_f_max):
        dx = 1e-6
        return (self.g_f(eps_f + dx, eps_f_max)[0] - self.g_f(eps_f, eps_f_max)[0]) / dx

    def get_corr_pred(self, eps, d_eps, sig, t_n, t_n1, alpha, q, kappa, eps_f_max):

        n_e, n_ip, n_s = eps.shape
        D = np.zeros((n_e, n_ip, 3, 3))

        # matrix
        sig[:,:, 0] = self.E_m*eps[:,:, 0]
        D[:,:, 0, 0] = self.E_m

        # reinforcement
        eps_f = eps[:,:, 2]
        sig_f_e = self.E_f * eps_f
        w_f, eps_f_max = self.g_f(eps_f, eps_f_max)
        print('w_f', np.amax(w_f))
        print('eps_f', np.amax(eps_f))
        sig[:,:, 2] = (1-w_f)*sig_f_e
        D[:,:, 2, 2] = -self.d_g_f(eps_f, eps_f_max) * sig_f_e + (1 - w_f) * self.E_f
#         sig[:,:, 2] = sig_f_e
#         D[:,:, 2, 2] = self.E_f

        # bond interface
        sig_trial = sig[:,:, 1]/(1-self.g(kappa)) + self.E_b * d_eps[:,:, 1]
        xi_trial = sig_trial - q
        f_trial = abs(xi_trial) - (self.sigma_y + self.K_bar * alpha)
        elas = f_trial <= 1e-8
        plas = f_trial > 1e-8
        d_gamma = f_trial / (self.E_b + self.K_bar + self.H_bar) * plas
        alpha += d_gamma
        kappa += d_gamma
        q += d_gamma * self.H_bar * np.sign(xi_trial)
        w = self.g(kappa)

        sig_e = sig_trial - d_gamma * self.E_b * np.sign(xi_trial)
        sig[:,:, 1] = (1-w)*sig_e

#         sig1[:, :, 1] = (1-w)*sig_e
#         print sig - sig1

        E_p = -self.E_b / (self.E_b + self.K_bar + self.H_bar) * derivative(self.g, kappa, dx=1e-6) * sig_e \
            + (1 - w) * self.E_b * (self.K_bar + self.H_bar) / \
            (self.E_b + self.K_bar + self.H_bar)

        D[:,:, 1, 1] = (1-w)*self.E_b*elas + E_p*plas

        return sig, D, alpha, q, kappa, eps_f_max

    n_s = Constant(3)


class FETS1D52ULRH(FETSEval):

    '''
    Fe Bar 2 nodes, deformation
    '''

    implements(IFETSEval)

    debug_on = True

    A_m = Float(100 * 8 - 9 * 1.85, desc='matrix area [mm2]')
    A_f = Float(9 * 1.85, desc='reinforcement area [mm2]')
    L_b = Float(9 * np.sqrt(np.pi * 4 * 1.85),
                desc='perimeter of the bond interface [mm]')

    # Dimensional mapping
    dim_slice = slice(0, 1)

    n_nodal_dofs = Int(2)

    dof_r = Array(value=[[-1], [1]])
    geo_r = Array(value=[[-1], [1]])
    vtk_r = Array(value=[[-1.], [1.]])
    vtk_cells = [[0, 1]]
    vtk_cell_types = 'Line'

    n_dof_r = Property
    '''Number of node positions associated with degrees of freedom. 
    '''
    @cached_property
    def _get_n_dof_r(self):
        return len(self.dof_r)

    n_e_dofs = Property
    '''Number of element degrees
    '''
    @cached_property
    def _get_n_dofs(self):
        return self.n_nodal_dofs * self.n_dof_r

    def _get_ip_coords(self):
        offset = 1e-6
        return np.array([[-1 + offset, 0., 0.], [1 - offset, 0., 0.]])

    def _get_ip_weights(self):
        return np.array([1., 1.], dtype=float)

    # Integration parameters
    #
    ngp_r = 2

    def get_N_geo_mtx(self, r_pnt):
        '''
        Return geometric shape functions
        @param r_pnt:
        '''
        r = r_pnt[0]
        N_mtx = np.array([[0.5 - r / 2., 0.5 + r / 2.]])
        return N_mtx

    def get_dNr_geo_mtx(self, r_pnt):
        '''
        Return the matrix of shape function derivatives.
        Used for the conrcution of the Jacobi matrix.
        '''
        return np.array([[-1. / 2, 1. / 2]])

    def get_N_mtx(self, r_pnt):
        '''
        Return shape functions
        @param r_pnt:local coordinates
        '''
        return self.get_N_geo_mtx(r_pnt)

    def get_dNr_mtx(self, r_pnt):
        '''
        Return the derivatives of the shape functions
        '''
        return self.get_dNr_geo_mtx(r_pnt)


class TStepper(HasTraits):

    '''Time stepper object for non-linear Newton-Raphson solver.
    '''

    mats_eval = Property(Instance(MATSEval))
    '''Finite element formulation object.
    '''
    @cached_property
    def _get_mats_eval(self):
        return MATSEval()

    fets_eval = Property(Instance(FETS1D52ULRH))
    '''Finite element formulation object.
    '''
    @cached_property
    def _get_fets_eval(self):
        return FETS1D52ULRH()

    A = Property()
    '''array containing the A_m, L_b, A_f
    '''
    @cached_property
    def _get_A(self):
        return np.array([self.fets_eval.A_m, self.fets_eval.L_b, self.fets_eval.A_f])

    domain = Property(Instance(FEGrid))
    '''Diescretization object.
    '''
    @cached_property
    def _get_domain(self):
        # Number of elements
        n_e_x = 60
        # length
        L_x = 300.
        # Element definition
        domain = FEGrid(coord_max=(L_x,),
                        shape=(n_e_x,),
                        fets_eval=self.fets_eval)
        return domain

    bc_list = List(Instance(BCDof))

    J_mtx = Property
    '''Array of Jacobian matrices.
    '''
    @cached_property
    def _get_J_mtx(self):
        fets_eval = self.fets_eval
        domain = self.domain
        # [ d, n ]
        geo_r = fets_eval.geo_r.T
        # [ d, n, i ]
        dNr_geo = geo_r[:,:, None] * np.array([1, 1]) * 0.5
        # [ i, n, d ]
        dNr_geo = np.einsum('dni->ind', dNr_geo)
        # [ n_e, n_geo_r, n_dim_geo ]
        elem_x_map = domain.elem_X_map
        # [ n_e, n_ip, n_dim_geo, n_dim_geo ]
        J_mtx = np.einsum('ind,enf->eidf', dNr_geo, elem_x_map)
        return J_mtx

    J_det = Property
    '''Array of Jacobi determinants.
    '''
    @cached_property
    def _get_J_det(self):
        return np.linalg.det(self.J_mtx)

    B = Property
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
        # [ d, n, i ]
        dNr_geo = geo_r[:,:, None] * np.array([1, 1]) * 0.5
        # [ i, n, d ]
        dNr_geo = np.einsum('dni->ind', dNr_geo)

        J_inv = np.linalg.inv(self.J_mtx)

        # shape function for the unknowns
        # [ d, n, i]
        Nr = 0.5 * (1. + geo_r[:,:, None] * r_ip[None,:])
        dNr = 0.5 * geo_r[:,:, None] * np.array([1, 1])

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
        B[:,:,:, B_N_n_rows, B_N_n_cols] = (B_factors[None, None,:] *
                                              Nx[:,:, N_idx])
        B[:,:,:, B_dN_n_rows, B_dN_n_cols] = dNx[:,:,:, dN_idx]

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

    def get_corr_pred(self, step_flag, U, d_U, eps, sig, t_n, t_n1, alpha, q, kappa, eps_f_max):
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

        # update strain
        eps += d_eps

        # material response state variables at integration point
        sig, D, alpha, q, kappa, eps_f_max = mats_eval.get_corr_pred(
            eps, d_eps, sig, t_n, t_n1, alpha, q, kappa, eps_f_max)

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
        return F_int, self.K, eps, sig, alpha, q, kappa, eps_f_max


class TLoop(HasTraits):

    ts = Instance(TStepper)
    d_t = Float(0.005)
    t_max = Float(2.0)
    k_max = Int(50)
    tolerance = Float(1e-5)

    def eval(self):

        self.ts.apply_essential_bc()

        t_n = 0.
        t_n1 = t_n
        n_dofs = self.ts.domain.n_dofs
        n_e = self.ts.domain.n_active_elems
        n_ip = self.ts.fets_eval.n_gp
        n_s = self.ts.mats_eval.n_s
        U_record = np.zeros(n_dofs)
        F_record = np.zeros(n_dofs)
        U_k = np.zeros(n_dofs)
        eps = np.zeros((n_e, n_ip, n_s))
        sig = np.zeros((n_e, n_ip, n_s))
        alpha = np.zeros((n_e, n_ip))
        q = np.zeros((n_e, n_ip))
        kappa = np.zeros((n_e, n_ip))
        eps_f_max = np.zeros((n_e, n_ip))

        while t_n1 <= self.t_max:
            t_n1 = t_n + self.d_t
            k = 0
            scale = 1.0
            step_flag = 'predictor'
            d_U = np.zeros(n_dofs)
            d_U_k = np.zeros(n_dofs)
            while k <= self.k_max:
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

                R, K, eps, sig, alpha, q, kappa, eps_f_max = self.ts.get_corr_pred(
                    step_flag, U_k, d_U_k, eps, sig, t_n, t_n1, alpha, q, kappa, eps_f_max)

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
#                     eps_r = eps
#                     sig_r = sig
#                     alpha_r = alpha
#                     q_r = q
#                     kappa_r = kappa
                    break
                k += 1
                step_flag = 'corrector'

            t_n = t_n1
        return U_record, F_record

if __name__ == '__main__':

    #=========================================================================
    # nonlinear solver
    #=========================================================================
    # initialization

    ts = TStepper()

    n_dofs = ts.domain.n_dofs

#     tf = lambda t: 1 - np.abs(t - 1)
#     ts.bc_list = [BCDof(var='u', dof=0, value=0.0),
# BCDof(var='u', dof=n_dofs - 1, value=2.5, time_function=tf)]

    ts.bc_list = [BCDof(var='u', dof=0, value=0.0),
                  BCDof(var='u', dof=n_dofs - 1, value=5.0)]

    tl = TLoop(ts=ts)

    U_record, F_record = tl.eval()
#     print 'U_record', U_record
    n_dof = 2 * ts.domain.n_active_elems + 1
#     print U_record[:, n_dof]
#     print F_record[:, n_dof]
    plt.plot(U_record[:, n_dof] * 2, F_record[:, n_dof] / 1000, marker='.')
    plt.ylim(0, 35)
    plt.xlabel('displacement')
    plt.ylabel('force')
    plt.show()

#     n_dofs = ts.domain.n_dofs
#
#     KMAX = 10
#     tolerance = 10e-4
#     U_k = np.zeros(n_dofs)
#
# time step parameters
#     t = 0.
#     tstep = 0.1
#     tmax = 1.0
#
# external force
#     F_ext = np.zeros(n_dofs)
#
# maximum force
#     n_e_x = ts.domain.shape[0]
#     F_max = np.zeros(n_dofs)
#     F_max[2 * n_e_x + 1] = 1.0
#
# for visualization
#     U_record = np.zeros(n_dofs)
#     F_record = np.zeros(n_dofs)
#
#     while t <= tmax:
#
#         t += tstep
#         F_ext += tstep * F_max
#
#         k = 0
#         step_flag = 'predictor'
#
#         while k < KMAX:
#
#             R, K = ts.get_corr_pred(U_k, t, F_ext, 0)
#
#             if np.linalg.norm(R) < tolerance:
#                 break
#
#             K.apply_constraints(R)
#             d_U = K.solve()
#
#             U_k += d_U
#             k += 1
#             step_flag = 'corrector'
#
#         U_record = np.vstack((U_record, U_k))
#         F_record = np.vstack((F_record, F_ext))
#
# print U_record[:, 5]
# print F_record[:, 5]
#     plt.plot(U_record[:, 5], F_record[:, 5], marker='o')
#     plt.xlabel('displacement')
#     plt.ylabel('force')
#     plt.show()
