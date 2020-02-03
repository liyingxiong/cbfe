
'''
Created on Sep 12, 2015

@author: rch
'''
from envisage.ui.workbench.api import WorkbenchApplication
from mayavi.sources.api import VTKDataSource, VTKFileReader

from ibvpy.fets.fets1D5 import FETS1D52L4ULRH
from ibvpy.mats.mats1D import MATS1DElastic
from ibvpy.mats.mats1D5.mats1D5_bond import MATS1D5Bond
from ibvpy.mesh.fe_grid import FEGrid
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly
import numpy as np

if __name__ == '__main__':

    #=========================================================================
    # Material matrix
    #=========================================================================
    mats_eval = MATS1D5Bond(mats_phase1=MATS1DElastic(E=10.),
                            mats_phase2=MATS1DElastic(E=20.),
                            mats_ifslip=MATS1DElastic(E=5.),
                            mats_ifopen=MATS1DElastic(E=1.))
    D_el = np.diag(np.array([10., 0.1, 1., 10.]))
    n_s = D_el.shape[0]

    #=========================================================================
    # Element definition
    #=========================================================================
    fets_eval = FETS1D52L4ULRH(mats_eval=mats_eval)
    n_geo_r, n_dim_geo = fets_eval.geo_r.shape
    n_dof_r, n_dim_dof = fets_eval.dof_r.shape
    n_ip = fets_eval.n_gp
    n_el_dofs = n_dof_r * n_dim_dof

    #[ d, i]
    r_ip = fets_eval.ip_coords[:, :-1].T
    # [ i ]
    w_ip = fets_eval.ip_weights.flatten()
    # [ d, n ]
    geo_r = fets_eval.geo_r.T
    # [ d, n, i ]
    dNr_geo = geo_r[
        :,:, None] * (1 + np.flipud(r_ip)[:, None,:] * np.flipud(geo_r)[:,:, None]) / 4.0
    # [ i, n, d ]
    dNr_geo = np.einsum('dni->ind', dNr_geo)

    #=========================================================================
    # Discretization
    #=========================================================================

    # Number of elements
    n_e_x = 1
    n_e_y = 1
    # length
    L_x = 20.0
    L_y = 2.
    # [ r, i ]
    domain = FEGrid(coord_max=(L_x, L_y),
                    shape=(n_e_x, n_e_y),
                    fets_eval=fets_eval)
    n_e = domain.n_active_elems
    n_dofs = domain.n_dofs
    # element array with nodal coordinates
    # [ n_e, n_geo_r, n_dim_geo ]
    elem_x_map = domain.elem_X_map
    print('elem_x_map', elem_x_map)
    # [ n_e, n_dof_r, n_dim_dof ]
    elem_dof_map = domain.elem_dof_map
#     print 'elem_dof_map', elem_dof_map

    # [ n_e, n_ip, n_dim_geo, n_dim_geo ]
    J_mtx = np.einsum('ind,enf->eidf', dNr_geo, elem_x_map)
    print('J_mtx', J_mtx.shape)
    J_inv = np.linalg.inv(J_mtx)
    print('J_inv', J_inv.shape)
    J_det = np.linalg.det(J_mtx)

    # shape function for the unknowns
    # [d, n, i]
    Nr = 0.5 * (1. + geo_r[None, 0,:, None]
                * r_ip[None, None, 0]) * np.array([[1], [ 1]])[:,:, None]
    dNr = 0.5 * geo_r[None, 0,:, None]*np.array([1, -1]) * np.array([[1], [ 1]])[:,:, None]    
    # [ i, n, d ]
    Nr = np.einsum('dni->ind', Nr)
    dNr = np.einsum('dni->ind', dNr)

    print(geo_r[0,:])
    print('Nr', Nr)
    print(Nr.shape)
    print('dNr', dNr)
    print(dNr.shape)

    # [ n_e, n_ip, n_dof_r, n_dim_dof ]
    dNx = np.einsum('eidf,inf->eind', J_inv, dNr)

    print('dNx', dNx.shape)
    print('dNx', dNx)

    B_N_factor = np.array([[1, 1],
                           [1, 1],
                           [-1, -1],
                           [-1, -1]])

    print(Nr[:,:, [0, 0]]*B_N_factor)

    B = np.zeros((n_e, n_ip, n_dof_r, n_s, n_dim_dof), dtype='f')
    B_N_rows, B_N_cols = [1, 2], [0, 1]
    B[:,:,:, B_N_rows, B_N_cols] = Nr[:,:, [0, 0]]*B_N_factor

    B[:,:, [0, 1], 0, 0] = dNx[:, 0, [0, 1], 0]
    B[:,:, [2, 3], 3, 0] = dNx[:, 0, [2, 3], 0]

    print(B)

#     B_n_rows, B_n_cols = [0, 1, 2], [0, 0, 1]
#     B[:,:, 0, B_n_rows, B_n_cols] = dNx[:,:, 0, 0],

    #=========================================================================
    # System matrix
    #=========================================================================
    K = np.einsum('i,einsd,st,eimtf,ei->endmf', w_ip, B, D_el, B, J_det)
    K_mtx = SysMtxAssembly()
    K_mtx.add_mtx_array(K.reshape(-1, n_el_dofs, n_el_dofs), elem_dof_map)

    print(K_mtx)

    #=========================================================================
    # Load vector
    #=========================================================================

    R = np.zeros((n_dofs,), dtype='float_')
    print(R)
    R[6] = 1.0
    K_mtx.register_constraint(a=0)
    K_mtx.register_constraint(a=1)
    K_mtx.register_constraint(a=2)
    u = K_mtx.solve(R)
    print('u', u)
