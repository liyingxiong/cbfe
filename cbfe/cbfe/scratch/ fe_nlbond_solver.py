from envisage.ui.workbench.api import WorkbenchApplication
from mayavi.sources.api import VTKDataSource, VTKFileReader

from traits.api import implements, Int, Array, HasTraits, Instance, \
    Property, cached_property, Constant, Float

from ibvpy.fets.fets_eval import FETSEval, IFETSEval
from ibvpy.mats.mats1D import MATS1DElastic
from ibvpy.mats.mats1D5.mats1D5_bond import MATS1D5Bond
from ibvpy.mesh.fe_grid import FEGrid
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly
import matplotlib.pyplot as plt
import numpy as np


class MATSEval(HasTraits):

    E_m = Float(10, tooltip='Stiffness of the matrix',
                auto_set=False, enter_set=False)

    E_f = Float(10, tooltip='Stiffness of the fiber',
                auto_set=False, enter_set=False)

#     G = Float(0.1, tooltip='Bond stiffness')
    def get_G(self, slip):
        return np.amax(0.2 - 0.2 * slip, 0.05)

    def get_D(self, eps, t, n_e, n_ip):
        D = np.zeros((n_e, n_ip, 3, 3))
        D[:,:, 0, 0] = self.E_m
        D[:,:, 2, 2] = self.E_f
        D[:,:, 1, 1] = self.get_G(eps[:,:, 1])
        print(eps[:,:, 1])
        print(self.get_G(eps[:,:, 1]))
        return D
#         return np.diag(np.array([self.E_m, self.G, self.E_f]))

    n_s = Constant(3)


class FETS1D52ULRH(FETSEval):

    '''
    Fe Bar 2 nodes, deformation
    '''

    implements(IFETSEval)

    debug_on = True

    # Dimensional mapping
    dim_slice = slice(0, 1)

    n_e_dofs = Int(4)
    n_nodal_dofs = Int(2)

    dof_r = Array(value=[[-1], [1]])
    geo_r = Array(value=[[-1], [1]])
    vtk_r = Array(value=[[-1.], [1.]])
    vtk_cells = [[0, 1]]
    vtk_cell_types = 'Line'

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

    domain = Property(Instance(FEGrid))
    '''Diescretization object.
    '''
    @cached_property
    def _get_domain(self):
        # Number of elements
        n_e_x = 10
        # length
        L_x = 20.0
        # Element definition
        domain = FEGrid(coord_max=(L_x,),
                        shape=(n_e_x,),
                        fets_eval=self.fets_eval)
        return domain

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

        n_dof_r, n_dim_dof = fets_eval.dof_r.shape
        n_dim_dof = 2
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

        B = np.zeros((n_e, n_ip, n_dof_r, n_s, n_dim_dof), dtype='f')
        B_N_n_rows, B_N_n_cols, N_idx = [1, 1], [0, 1], [0, 0]
        B_dN_n_rows, B_dN_n_cols, dN_idx = [0, 2], [0, 1], [0, 0]
        B_factors = np.array([-1, 1], dtype='float_')
        B[:,:,:, B_N_n_rows, B_N_n_cols] = (B_factors[None, None,:] *
                                              Nx[:,:, N_idx])
        B[:,:,:, B_dN_n_rows, B_dN_n_cols] = dNx[:,:,:, dN_idx]

        return B

    def apply_constraints(self, K_mtx, F_ext):
        '''Insert boundary conditions into the system matrix and rhs vector. 
        '''
        n_dofs = self.domain.n_dofs

        R = np.zeros((n_dofs,), dtype='float_')

        n_e_x = self.domain.shape[0]
        R[2 * n_e_x + 1] = 1.0
        K_mtx.register_constraint(a=0)

    def get_corr_pred(self, U, d_U, t, F_ext, fixed, F_inter, eps):
        # parameters
        mats_eval = self.mats_eval
        fets_eval = self.fets_eval
        domain = self.domain
        elem_dof_map = domain.elem_dof_map
        n_e = domain.n_active_elems
        n_dof_r, n_dim_dof = self.fets_eval.dof_r.shape
        n_dim_dof = 2
        n_dofs = domain.n_dofs
        J_det = np.linalg.det(self.J_mtx)
        w_ip = fets_eval.ip_weights
        n_el_dofs = n_dof_r * n_dim_dof
        n_ip = self.fets_eval.n_gp

        # evaluate internal force increment
        D = mats_eval.get_D(eps, t, n_e, n_ip)
        # element displacement increment
        d_u_e = d_U[elem_dof_map]
        #[n_e, n_dof_r, n_dim_dof]
        d_u_n = d_u_e.reshape(n_e, n_dof_r, n_dim_dof)
        #[n_e, n_ip, n_s]
        d_eps = np.einsum('einsd,end->eis', self.B, d_u_n)
        # stress increment
        #[n_e, n_ip, n_s]
        d_sig = np.einsum('eist,eit->eis', D, d_eps)
        # internal force increment
        # [n_e, n_n, n_dim_dof]
        d_f_inter_e = np.einsum('eis,einsd,ei->end', d_sig, self.B, J_det)
        d_f_inter_e = d_f_inter_e.reshape(n_e, n_dof_r * n_dim_dof)
        d_F_inter = np.zeros(n_dofs)
        np.add.at(d_F_inter, elem_dof_map, d_f_inter_e)
        # fixed dof
        d_F_inter[fixed] = 0.
        # update internal force
        F_inter += d_F_inter

        # update strain and D matrix
        eps += d_eps
        D = mats_eval.get_D(eps, t, n_e, n_ip)
        # update stiffness matrix
        K = np.einsum('i,einsd,eist,eimtf,ei->endmf',
                      w_ip, self.B, D, self.B, J_det)
        K_mtx = SysMtxAssembly()
        K_mtx.add_mtx_array(K.reshape(-1, n_el_dofs, n_el_dofs), elem_dof_map)
        K_mtx.register_constraint(a=fixed)

        return F_ext - F_inter, K_mtx, F_inter, eps


class TLoop(HasTraits):

    ts = Instance(TStepper)
    tstep = Float(0.05)
    t_max = Float(1.0)
    k_max = Int(100)
    F_max = Array
    tolerance = Float(1e-4)

    def eval(self):
        t = 0.
        n_dofs = self.ts.domain.n_dofs
        n_e = self.ts.domain.n_active_elems
        n_ip = self.ts.fets_eval.n_gp
        n_s = self.ts.mats_eval.n_s
        U_record = np.zeros(n_dofs)
        F_record = np.zeros(n_dofs)
        F_ext = np.zeros(n_dofs)
        F_inter = np.zeros(n_dofs)
        U_k = np.zeros(n_dofs)
        eps = np.zeros((n_e, n_ip, n_s))

        while t <= self.t_max:
            t += self.tstep
            F_ext += self.tstep * self.F_max
            k = 0
            d_U = np.zeros(n_dofs)
            step_flag = 'predictor'
            while k < self.k_max:
                R, K, F_inter, eps = ts.get_corr_pred(
                    U_k, d_U, t, F_ext, 0, F_inter, eps)
                if np.linalg.norm(R) < self.tolerance:
                    break
                K.apply_constraints(R)
                d_U = K.solve()
                U_k += d_U
                k += 1
                step_flag = 'corrector'
            U_record = np.vstack((U_record, U_k))
            F_record = np.vstack((F_record, F_ext))
        return U_record, F_record

if __name__ == '__main__':

    #=========================================================================
    # nonlinear solver
    #=========================================================================
    # initialization

    ts = TStepper()

    n_dofs = ts.domain.n_dofs
    F_max = np.zeros(n_dofs)
    F_max[-1] = 1.0
    tl = TLoop(ts=ts,
               F_max=F_max)
    U_record, F_record = tl.eval()
    plt.plot(U_record[:, 5], F_record[:, 5], marker='o')
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
