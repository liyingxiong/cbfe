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
from scipy.interpolate import interp1d


class MATSEval(HasTraits):

    E_m = Float(28484, tooltip='Stiffness of the matrix',
                auto_set=False, enter_set=False)

    E_f = Float(170000, tooltip='Stiffness of the fiber',
                auto_set=False, enter_set=False)

    # 30-v1g-r3-f
#     slip = List([0.0, 0.38461538461538458, 0.89743589743589736, 1.4102564102564101, 1.9230769230769229, 2.4358974358974357, 2.9487179487179485, 3.4615384615384612, 3.974358974358974, 4.4871794871794872,
#                  5.0, 5.5128205128205128, 6.0256410256410255, 6.5384615384615383, 7.0512820512820511, 7.5641025641025639, 8.0769230769230766, 8.5897435897435876, 9.1025641025641022, 9.6153846153846132, 10.0])
#     bond = List([0.0, 41.131553809206011, 39.936804665067541, 43.088224620622753, 47.650913085297518, 52.479625415305165, 57.394429686101795, 62.251107904045313, 66.738355595565764, 70.767647986595335, 73.873975043302551,
# 76.009215625340516, 77.277633801765163, 76.270142784043799,
# 73.416598577944114, 69.69994680116541, 64.730320409930115,
# 58.933638345787244, 52.554181600645506, 45.626326986883868,
# 39.96573453999463])

    # 30-v2-r3-f
#     slip = List([0.0, 0.055714285714285716, 0.13, 0.20428571428571429, 0.40500000000000003, 0.95862068965517244, 1.5931034482758619, 2.2275862068965515, 2.8620689655172411, 3.4965517241379307,
#                  4.1310344827586203, 4.7655172413793103, 5.3999999999999986, 6.0344827586206886, 6.6689655172413786, 7.3034482758620687, 7.9379310344827569, 8.572413793103447, 9.2068965517241388, 9.841379310344827])
#     bond = List([0.0, 15.00385088356251, 35.008985394979184, 55.014119906395834, 43.133431746905352, 42.22129154275072, 46.457189937422498, 52.421832224015986, 58.214528732629347, 63.443110953129597,
# 65.081745830172082, 70.818555581926134, 76.171908502739569,
# 80.423245044030807, 83.434389826392007, 85.024361743793122,
# 82.174874390870798, 76.343148188571817, 66.095308161727061,
# 54.526649351153921])

    # 30-v3-r3-f
#     slip = List([0.0, 0.061874999999999999, 0.144375, 0.22687499999999999, 0.30937500000000001, 0.72500000000000009, 1.2758620689655173, 1.9103448275862067, 2.5448275862068965, 3.1793103448275857,
#                  3.8137931034482757, 4.4482758620689644, 5.0827586206896544, 5.7172413793103445, 6.3517241379310345, 6.9862068965517228, 7.6206896551724128, 8.2551724137931028, 8.8896551724137929, 9.5241379310344811, 10.0])
#     bond = List([0.0, 16.27428330083729, 37.973327701953664, 59.672372103070117, 43.400401511742785, 43.234649080448378, 48.701090597581455, 54.736977654692424, 62.048663399413229, 69.370097603432924, 77.083431690155379,
# 83.982393896019687, 88.794804003264773, 92.364321409231025,
# 92.988748502860574, 90.587882722159577, 85.05512495854191,
# 80.343909075206099, 75.103929360931232, 66.591754226257905,
# 62.36344706176229])

    # 20-v1-r3-f
#     slip = List([0.0, 0.029062499999999998, 0.067812499999999998, 0.1065625, 0.14531250000000001, 0.29999999999999999, 0.75, 1.34375, 1.9354166666666668, 2.5270833333333336,
#                  3.1187499999999999, 3.7104166666666667, 4.3020833333333339, 4.8937500000000007, 5.4854166666666675, 6.0770833333333343, 6.6687500000000011, 7.2604166666666679, 7.8520833333333337])
#     bond = List([0.0, 16.539040266437738, 38.591093955021357, 60.643147643604991, 55.492896981222152, 33.361147503574301, 41.750737025152148, 45.341604537187422, 53.984990428825952, 62.276659123066786,
# 70.624931767476426, 77.79727324562819, 82.881005240210555,
# 83.82596383699105, 85.775479050115678, 85.647895874781625,
# 82.121406801790954, 78.787654614781076, 74.308361153830276])

    # 20-v2-r3-f
#     slip = List([0.0, 0.025312500000000002, 0.059062500000000004, 0.092812500000000006, 0.12656250000000002, 0.28999999999999998, 0.75, 1.34375, 1.9354166666666668, 2.5270833333333336,
#                  3.1187499999999999, 3.7104166666666667, 4.3020833333333339, 4.8937500000000007, 5.4854166666666675, 6.0770833333333343, 6.6687500000000011, 7.2604166666666679, 7.8520833333333337])
#     bond = List([0.0, 16.497410959806412, 38.493958906214978, 60.490506852623582, 53.730724862691076, 32.368976701926144, 46.320380964466182, 53.362551389229182, 60.895571700733498, 70.421645193646242,
# 81.283769953964168, 88.218274137431948, 92.648027453513293,
# 97.858499164610748, 99.266642697732422, 91.860288624493535,
# 78.662911574914844, 64.495097443305497, 53.318329359069025])

    # 30-v1g
#     slip = List([0.0, 0.3125, 0.72916666666666674, 1.1458333333333335, 1.5625, 1.9791666666666667, 2.3958333333333335,
#                  2.8125, 3.229166666666667, 3.6458333333333335, 4.0625, 4.479166666666667, 4.8958333333333339])
#     bond = List([0.0, 36.778418900302242, 39.758442442202977, 40.884882306336046, 44.445217145181815, 48.118479636057231,
# 52.070269352966747, 56.040212250977582, 60.060774337096234,
# 63.945337675621374, 67.39679786075925, 70.738985619426003,
# 73.401606879129417])
    slip = List
    bond = List

    def b_s_law(self, x):
        return np.interp(x, self.slip, self.bond)

    def G(self, x):
        d = np.diff(self.bond) / np.diff(self.slip)
        d = np.append(d, d[-1])
        G = interp1d(np.array(self.slip), d, kind='zero')
        y = np.zeros_like(x)
        y[x < self.slip[0]] = d[0]
        y[x > self.slip[-1]] = d[-1]
        x[x < self.slip[0]] = self.slip[-1] + 10000.
        y[x <= self.slip[-1]] = G(x[x <= self.slip[-1]])
        return y

    def get_corr_pred(self, eps, d_eps, sig, t_n, t_n1):
        n_e, n_ip, n_s = eps.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[:, :, 0, 0] = self.E_m
        D[:, :, 2, 2] = self.E_f
        D[:, :, 1, 1] = self.G(eps[:,:, 1])
        d_sig = np.einsum('...st,...t->...s', D, d_eps)
        sig += d_sig
        sig[:, :, 1] = self.b_s_law(eps[:,:, 1])
        return sig, D

    n_s = Constant(3)


class FETS1D52ULRH(FETSEval):

    '''
    Fe Bar 2 nodes, deformation
    '''

    implements(IFETSEval)

    debug_on = True

    A_m = Float(120 * 13 - 9 * 1.85, desc='matrix area [mm2]')
    A_f = Float(9 * 1.85, desc='reinforcement area [mm2]')
    L_b = Float(1., desc='perimeter of the bond interface [mm]')

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

    # number of elements
    n_e_x = Float(20.)

    # specimen length
    L_x = Float(75.)

    domain = Property(depends_on='n_e_x, L_x')
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

    J_mtx = Property(depends_on='n_e_x, L_x')
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

    J_det = Property(depends_on='n_e_x, L_x')
    '''Array of Jacobi determinants.
    '''
    @cached_property
    def _get_J_det(self):
        return np.linalg.det(self.J_mtx)

    B = Property(depends_on='n_e_x, L_x')
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
        dNr_geo = geo_r[:, :, None] * np.array([1, 1]) * 0.5
        # [ i, n, d ]
        dNr_geo = np.einsum('dni->ind', dNr_geo)

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

    def get_corr_pred(self, step_flag, d_U, eps, sig, t_n, t_n1):
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
#         if np.any(sig) == np.nan:
#             sys.exit()

        # material response state variables at integration point
        sig, D = mats_eval.get_corr_pred(eps, d_eps, sig, t_n, t_n1)

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
        return F_int, self.K, eps, sig


class TLoop(HasTraits):

    ts = Instance(TStepper)
    d_t = Float(0.01)
    t_max = Float(1.0)
    k_max = Int(200)
    tolerance = Float(1e-8)

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

        while t_n1 <= self.t_max:
            t_n1 = t_n + self.d_t
            print(t_n1)
            k = 0
            step_flag = 'predictor'
            d_U = np.zeros(n_dofs)
            d_U_k = np.zeros(n_dofs)
            while k < self.k_max:
                R, K, eps, sig = self.ts.get_corr_pred(
                    step_flag, d_U_k, eps, sig, t_n, t_n1)

                F_ext = -R
                K.apply_constraints(R)
                d_U_k = K.solve()
                d_U += d_U_k
                if np.linalg.norm(R) < self.tolerance:
                    F_record = np.vstack((F_record, F_ext))
                    U_k += d_U
                    U_record = np.vstack((U_record, U_k))
                    break
                k += 1
                if k == self.k_max:
                    print('nonconvergence')
                step_flag = 'corrector'

            t_n = t_n1
        return U_record, F_record

if __name__ == '__main__':

    #=========================================================================
    # nonlinear solver
    #=========================================================================
    # initialization

    ts = TStepper()

#     x, y = np.loadtxt('D:\\bondlaw.txt')
#
    ts.mats_eval.slip = [0.0, 0.09375, 0.505, 0.90172413793103456, 1.2506896551724138, 1.5996551724137933, 1.9486206896551728, 2.2975862068965522, 2.6465517241379315,
                         2.9955172413793107, 3.34448275862069, 3.6934482758620693, 4.0424137931034485, 4.3913793103448278, 4.7403448275862079, 5.0893103448275863, 5.4382758620689664, 5.7000000000000002]

    ts.mats_eval.bond = [0.0, 43.05618551913318, 40.888629416715574, 49.321970730383285, 56.158143245133338, 62.245706611484323, 68.251000923721875, 73.545464379399633, 79.032738465995692,
                         84.188949455670524, 87.531858162376921, 91.532666285021264, 96.66808302759236, 100.23305856244875, 103.01090365681807, 103.98920712455558, 104.69444418370917, 105.09318577617957]

    n_dofs = ts.domain.n_dofs

#     tf = lambda t: 1 - np.abs(t - 1)

    ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),
                  BCDof(var='u', dof=n_dofs - 1, value=6.0)]

    tl = TLoop(ts=ts)

    U_record, F_record = tl.eval()
    n_dof = 2 * ts.domain.n_active_elems + 1
#     x, y = np.loadtxt('D:\\1.txt')
#     plt.plot(x, y)
    plt.plot(U_record[:, n_dof], F_record[:, n_dof],
             marker='.', label='numerical')

    fpath = 'D:\\no15.csv'
    x, y = np.loadtxt(fpath,  delimiter=',').T
    plt.plot(x / 2, y * 1000)

#     fpath = 'D:\\data\\pull_out\\all\\DPO-40cm-0-3300SBR-V3_R3_f.asc'
#     x, y = np.loadtxt(fpath,  delimiter=';')
#     plt.plot(x / 2., y * 1000., 'k--', label='experimental')

#     fpath = 'D:\\data\\pull_out\\all\\DPO-40cm-0-3300SBR-V2_R3_f.asc'
#     x, y = np.loadtxt(fpath,  delimiter=';')
#     plt.plot(x / 2., y * 1000., 'k--')
#
#     fpath = 'D:\\data\\pull_out\\all\\DPO-40cm-0-3300SBR-V3_R3_f.asc'
#     x, y = np.loadtxt(fpath,  delimiter=';')
#     plt.plot(x / 2., y * 1000., 'k--')

    plt.xlabel('displacement [mm]')
    plt.ylabel('pull-out force [N]')
    plt.ylim(0, 20000)
    plt.legend(loc='best')

    plt.show()
