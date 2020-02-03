
'''
Created on Sep 12, 2015

@author: rch
'''
from envisage.ui.workbench.api import WorkbenchApplication
from mayavi.sources.api import VTKDataSource, VTKFileReader
from traits.api import implements, Int, Array
from ibvpy.fets.fets_eval import FETSEval, IFETSEval
from ibvpy.mats.mats1D import MATS1DElastic
from ibvpy.mats.mats1D5.mats1D5_bond import MATS1D5Bond
from ibvpy.mesh.fe_grid import FEGrid
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':

    #=========================================================================
    # Material matrix
    #=========================================================================
    D_el = np.diag(np.array([10., 0.1, 10.]))
    n_s = D_el.shape[0]

    #=========================================================================
    # Element definition
    #=========================================================================
    fets_eval = FETS1D52ULRH()
    n_geo_r, n_dim_geo = fets_eval.geo_r.shape
    n_dof_r, n_dim_dof = fets_eval.dof_r.shape
    n_dim_dof = 2
    n_ip = fets_eval.n_gp
    n_el_dofs = n_dof_r * n_dim_dof

    #[ d, i]
    r_ip = fets_eval.ip_coords[:, :-2].T
    # [ i ]
    w_ip = fets_eval.ip_weights
    # [ d, n ]
    geo_r = fets_eval.geo_r.T
    # [ d, n, i ]
    dNr_geo = geo_r[:, :, None] * np.array([1, 1]) * 0.5
    # [ i, n, d ]
    dNr_geo = np.einsum('dni->ind', dNr_geo)

    #=========================================================================
    # Discretization
    #=========================================================================

    # Number of elements
    n_e_x = 10
    # length
    L_x = 20.0
    # [ r, i ]
    domain = FEGrid(coord_max=(L_x,),
                    shape=(n_e_x,),
                    fets_eval=fets_eval)
    n_e = domain.n_active_elems
    n_dofs = domain.n_dofs
    # element array with nodal coordinates
    # [ n_e, n_geo_r, n_dim_geo ]
    elem_x_map = domain.elem_X_map
#     print elem_x_map
    # [ n_e, n_dof_r, n_dim_dof ]
    elem_dof_map = domain.elem_dof_map
#     print 'elem_dof_map', elem_dof_map

    # [ n_e, n_ip, n_dim_geo, n_dim_geo ]
    J_mtx = np.einsum('ind,enf->eidf', dNr_geo, elem_x_map)
    J_inv = np.linalg.inv(J_mtx)
    J_det = np.linalg.det(J_mtx)

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

    B = np.zeros((n_e, n_ip, n_dof_r, n_s, n_dim_dof), dtype='f')
    B_N_n_rows, B_N_n_cols, N_idx = [1, 1], [0, 1], [0, 0]
    B_dN_n_rows, B_dN_n_cols, dN_idx = [0, 2], [0, 1], [0, 0]
    B_factors = np.array([-1, 1], dtype='float_')
    B[:, :,:, B_N_n_rows, B_N_n_cols] = (B_factors[None, None,:] *
                                          Nx[:, :, N_idx])
    B[:, :,:, B_dN_n_rows, B_dN_n_cols] = dNx[:,:,:, dN_idx]

    #=========================================================================
    # initialization
    #=========================================================================
    # material matrix
    D_el = np.zeros((n_e, n_ip, 3, 3))
    D_el[:, :, 0, 0] = 10.
    D_el[:, :, 2, 2] = 10.
    D_el[:, :, 1, 1] = 0.001

    u_norm = 10

    u_old = np.zeros(n_dofs)

    while u_norm >= 0.001:

        #======================================================================
        # System matrix
        #======================================================================
        K = np.einsum('i,einsd,eist,eimtf,ei->endmf', w_ip, B, D_el, B, J_det)
        K_mtx = SysMtxAssembly()
        K_mtx.add_mtx_array(K.reshape(-1, n_el_dofs, n_el_dofs), elem_dof_map)

        #======================================================================
        # BC and solver
        #======================================================================
        R = np.zeros((n_dofs,), dtype='float_')
    #     R[2 * n_e_x + 1] = 1.0
        K_mtx.register_constraint(a=0)
        K_mtx.register_constraint(a=2 * n_e_x + 1, u_a=1.2)
        u = K_mtx.solve(R)
        print('u', u)

        #======================================================================
        # strain and slip
        #======================================================================
        u_e = u[elem_dof_map]
        #[n_e, n_dof_r, n_dim_dof]
        u_n = u_e.reshape(n_e, n_dof_r, n_dim_dof)
        #[n_e, n_ip, n_s]
        eps = np.einsum('einsd,end->eis', B, u_n)

        #======================================================================
        # bond slip relationship
        #======================================================================
        slip = eps[:, :, 1]
        bond = np.piecewise(slip, [slip < 1., (1 <= slip) & (slip <= 1.5), slip > 1.5],
                            [lambda s: 0.2 * s, lambda s: 0.4 - 0.2 * s, 0.1])

        #======================================================================
        # update D_el
        #======================================================================
        D_el[:, :, 1, 1] = bond

        #======================================================================
        # for convergence check
        #======================================================================
        u_norm = np.linalg.norm(u - u_old)
        print(u_norm)
        u_old = u

    #=========================================================================
    # stress and shear flow
    #=========================================================================
    #[n_e, n_ip, n_s]
#     sig = eps * np.diag(D_el)

    #=========================================================================
    # visualization
    #=========================================================================
    # deformation
    deform = plt.subplot(421)
    n_coord = np.unique(elem_x_map)
    plt.plot(n_coord + u[0::2], np.zeros_like(u[0::2]), 'b', marker='.')
    plt.plot(n_coord + u[1::2], np.ones_like(u[0::2]), 'g', marker='.')
    plt.ylim((-1, 2))
    deform.axes.get_yaxis().set_visible(False)
    deform.set_title('deformation')

    # coordinates of the integration points
    ip_coord = np.einsum('end,ind->ei', elem_x_map, Nr)
    ip_coord = ip_coord.flatten()

    # matrix strain
    m_strain = plt.subplot(423)
    plt.plot(ip_coord, eps[:, :, 0].flatten(), 'b')
    m_strain.set_title('matrix strain')

    # fiber strain
    f_strain = plt.subplot(425)
    plt.plot(ip_coord, eps[:, :, 2].flatten(), 'g')
    f_strain.set_title('fiber strain')

    # slip
    slip = plt.subplot(427)
    plt.plot(ip_coord, eps[:, :, 1].flatten(), 'k')
    slip.set_title('slip')

# matrix stress
#     m_stress = plt.subplot(424)
#     plt.plot(ip_coord, sig[:, :, 0].flatten(), 'b')
#     m_stress.set_title('matrix stress')
#
# fiber stress
#     f_stress = plt.subplot(426)
#     plt.plot(ip_coord, sig[:, :, 2].flatten(), 'g')
#     f_stress.set_title('fiber stress')
#
# shear
#     shear = plt.subplot(428)
#     plt.plot(ip_coord, sig[:, :, 1].flatten(), 'K')
#     shear.set_title('shear flow')

    plt.subplots_adjust(bottom=0.05, top=0.95, hspace=0.25)
    plt.show()
