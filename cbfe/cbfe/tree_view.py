'''
Created on 12.01.2016

@author: Yingxiong
'''
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from traits.api import \
    HasTraits, Property, Instance, cached_property, Str, Button, \
    Range, on_trait_change, Array, List, Float
from traitsui.api import \
    View, Item, Group, VGroup, HSplit, TreeEditor, TreeNode

from .fets1d52ulrh import FETS1D52ULRH
from ibvpy.api import BCDof
from .matseval import MATSEval
import numpy as np
from .tloop import TLoop
from .tstepper import TStepper
from util.traits.editors import MPLFigureEditor


class Material(HasTraits):
    sigma_y = Range(0.5, 55, value=1.05)
    E_b = Range(1.0, 80, value=2.0)
    K_bar = Range(-0.01, 0.15, value=0.08)
    H_bar = Range(-0.1, 0.1, value=0.00)
    alpha = Range(0.50, 2.50, value=1.0)
    beta = Range(0.00, 1.00, value=1.0)


class Geometry(HasTraits):
    L_x = Range(100, 400, value=300)
    A_m = Float(100 * 8 - 9 * 1.85, desc='matrix area [mm2]')
    A_f = Float(9 * 1.85, desc='reinforcement area [mm2]')
    P_b = Float(9 * np.sqrt(np.pi * 4 * 1.85),
                desc='perimeter of the bond interface [mm]')


class NSolver(HasTraits):
    d_t = Float(0.01)
    t_max = Float(1.)
    k_max = Float(50)
    tolerance = Float(1e-5)
    disps = Str('0.0,10.0')

    d_array = Property(depends_on='disps')
    ''' convert the disps string to float array
    '''
    @cached_property
    def _get_d_array(self):
        return np.array([float(x) for x in self.disps.split(',')])

    time_func = Property(depends_on='disps, t_max')

    @cached_property
    def _get_time_func(self):
        dd_arr = np.abs(np.diff(self.d_array))
        x = np.hstack((0, self.t_max * np.cumsum(dd_arr) / sum(dd_arr)))
        return interp1d(x, self.d_array)

    figure = Instance(Figure)

    def _figure_default(self):
        figure = Figure()
        return figure

    update = Button()

    def _update_fired(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        x = np.arange(0, self.t_max, self.d_t)
        ax.plot(x, self.time_func(x))
        ax.set_xlabel('time')
        ax.set_ylabel('displacement')
        self.figure.canvas.draw()

    view = View(HSplit(Group(Item('d_t'),
                             Item('k_max')),
                       Group(Item('t_max'),
                             Item('tolerance'))),
                Group(Item('disps'),
                      Item('update', show_label=False)),
                Item('figure', editor=MPLFigureEditor(),
                     dock='horizontal', show_label=False),
                kind='modal')


class TreeStructure(HasTraits):

    material = List(value=[Material()])
    geometry = List(value=[Geometry()])
    n_solver = List(value=[NSolver()])
    name = Str('pull out simulation')


class MainWindow(HasTraits):

    mats_eval = Instance(MATSEval)

    fets_eval = Instance(FETS1D52ULRH)

    time_stepper = Instance(TStepper)

    time_loop = Instance(TLoop)

    tree = Instance(TreeStructure)

    t_record = Array
    U_record = Array
    F_record = Array
    sf_record = Array
    eps_record = List
    sig_record = List

    figure = Instance(Figure)

    def _figure_default(self):
        figure = Figure()
        return figure

    plot = Button()

    def _plot_fired(self):
        # assign the material parameters
        self.mats_eval.sigma_y = self.tree.material[0].sigma_y
        self.mats_eval.E_b = self.tree.material[0].E_b
        self.mats_eval.K_bar = self.tree.material[0].K_bar
        self.mats_eval.alpha = self.tree.material[0].alpha
        self.mats_eval.beta = self.tree.material[0].beta
        self.mats_eval.H_bar = self.tree.material[0].H_bar

        # assign the geometry parameters
        self.fets_eval.A_m = self.tree.geometry[0].A_m
        self.fets_eval.P_b = self.tree.geometry[0].P_b
        self.fets_eval.A_f = self.tree.geometry[0].A_f
        self.time_stepper.L_x = self.tree.geometry[0].L_x

        # assign the parameters for solver
        self.time_loop.t_max = self.tree.n_solver[0].t_max
        self.time_loop.d_t = self.tree.n_solver[0].d_t
        self.time_loop.k_max = self.tree.n_solver[0].k_max
        self.time_loop.tolerance = self.tree.n_solver[0].tolerance

        # assign the bc
        self.time_stepper.bc_list[1].value = 1.0
        self.time_stepper.bc_list[
            1].time_function = self.tree.n_solver[0].time_func

        self.draw()
        self.time = 1.00
#         self.figure.canvas.draw()

    ax1 = Property()

    @cached_property
    def _get_ax1(self):
        return self.figure.add_subplot(231)

    ax2 = Property()

    @cached_property
    def _get_ax2(self):
        return self.figure.add_subplot(232)

    ax3 = Property()

    @cached_property
    def _get_ax3(self):
        return self.figure.add_subplot(234)

    ax4 = Property()

    @cached_property
    def _get_ax4(self):
        return self.figure.add_subplot(235)

    ax5 = Property()

    @cached_property
    def _get_ax5(self):
        return self.figure.add_subplot(233)

    ax6 = Property()

    @cached_property
    def _get_ax6(self):
        return self.figure.add_subplot(236)

    def draw(self):
        self.U_record, self.F_record, self.sf_record, self.t_record, self.eps_record, self.sig_record = self.time_loop.eval()
        n_dof = 2 * self.time_stepper.domain.n_active_elems + 1

        slip, sig_n_arr, sig_e_arr, w_arr = self.time_stepper.mats_eval.get_bond_slip()
        self.ax1.cla()
        l_bs, = self.ax1.plot(slip, sig_n_arr)
        self.ax1.plot(slip, sig_e_arr, '--')
        self.ax1.plot(slip, w_arr, '--')
        self.ax1.set_title('bond-slip law')

        self.ax2.cla()
        l_po, = self.ax2.plot(self.U_record[:, n_dof], self.F_record[:, n_dof])
        marker_po, = self.ax2.plot(
            self.U_record[-1, n_dof], self.F_record[-1, n_dof], 'ro')
        self.ax2.set_title('pull-out force-displacement curve')

        self.ax3.cla()
        X = np.linspace(
            0, self.time_stepper.L_x, self.time_stepper.n_e_x + 1)
        X_ip = np.repeat(X, 2)[1:-1]
        l_sf, = self.ax3.plot(X_ip, self.sf_record[-1, :])
        self.ax3.set_title('shear flow in the bond interface')

        self.ax4.cla()
        U = np.reshape(self.U_record[-1, :], (-1, 2)).T
        l_u0, = self.ax4.plot(X, U[0])
        l_u1, = self.ax4.plot(X, U[1])
        l_us, = self.ax4.plot(X, U[1] - U[0])
        self.ax4.set_title('displacement and slip')

        self.ax5.cla()
        l_eps0, = self.ax5.plot(X_ip, self.eps_record[-1][:, :, 0].flatten())
        l_eps1, = self.ax5.plot(X_ip, self.eps_record[-1][:, :, 2].flatten())
        self.ax5.set_title('strain')

        self.ax6.cla()
        l_sig0, = self.ax6.plot(X_ip, self.sig_record[-1][:, :, 0].flatten())
        l_sig1, = self.ax6.plot(X_ip, self.sig_record[-1][:, :, 2].flatten())
        self.ax6.set_title('stress')

        self.ax3.set_ylim(np.amin(self.sf_record), np.amax(self.sf_record))
        self.ax4.set_ylim(np.amin(self.U_record), np.amax(self.U_record))
        self.ax6.set_ylim(np.amin(self.sig_record), np.amax(self.sig_record))

        self.figure.canvas.draw()

    time = Range(0.00, 1.00, value=1.00)

    @on_trait_change('time')
    def draw_t(self):
        idx = (np.abs(self.time * max(self.t_record) - self.t_record)).argmin()
        n_dof = 2 * self.time_stepper.domain.n_active_elems + 1

        self.ax2.cla()
        l_po, = self.ax2.plot(self.U_record[:, n_dof], self.F_record[:, n_dof])
        marker_po, = self.ax2.plot(
            self.U_record[idx, n_dof], self.F_record[idx, n_dof], 'ro')
        self.ax2.set_title('pull-out force-displacement curve')

        self.ax3.cla()
        X = np.linspace(
            0, self.time_stepper.L_x, self.time_stepper.n_e_x + 1)
        X_ip = np.repeat(X, 2)[1:-1]
        l_sf, = self.ax3.plot(X_ip, self.sf_record[idx, :])
        self.ax3.set_title('shear flow in the bond interface')

        self.ax4.cla()
        U = np.reshape(self.U_record[idx, :], (-1, 2)).T
        l_u0, = self.ax4.plot(X, U[0])
        l_u1, = self.ax4.plot(X, U[1])
        l_us, = self.ax4.plot(X, U[1] - U[0])
        self.ax4.set_title('displacement and slip')

        self.ax5.cla()
        l_eps0, = self.ax5.plot(X_ip, self.eps_record[idx][:, :, 0].flatten())
        l_eps1, = self.ax5.plot(X_ip, self.eps_record[idx][:, :, 2].flatten())
        self.ax5.set_title('strain')

        self.ax6.cla()
        l_sig0, = self.ax6.plot(X_ip, self.sig_record[idx][:, :, 0].flatten())
        l_sig1, = self.ax6.plot(X_ip, self.sig_record[idx][:, :, 2].flatten())
        self.ax6.set_title('stress')

        self.ax3.set_ylim(np.amin(self.sf_record), np.amax(self.sf_record))
        self.ax4.set_ylim(np.amin(self.U_record), np.amax(self.U_record))
        self.ax6.set_ylim(np.amin(self.sig_record), np.amax(self.sig_record))

        self.figure.canvas.draw()

    tree_editor = TreeEditor(nodes=[TreeNode(node_for=[TreeStructure],
                                             children='',
                                             label='name',
                                             view=View()),
                                    TreeNode(node_for=[TreeStructure],
                                             children='material',
                                             label='=Material',
                                             auto_open=True,
                                             view=View()),
                                    TreeNode(node_for=[TreeStructure],
                                             children='geometry',
                                             label='=Geometry',
                                             auto_open=True,
                                             view=View()),
                                    TreeNode(node_for=[TreeStructure],
                                             children='n_solver',
                                             label='=Nonlinear Solver',
                                             auto_open=True,
                                             view=View()),
                                    TreeNode(node_for=[Material],
                                             children='',
                                             label='=Material parameters'),
                                    TreeNode(node_for=[Geometry],
                                             children='',
                                             label='=Geometry parameters'),
                                    TreeNode(node_for=[NSolver],
                                             children='',
                                             label='=settings')
                                    ],
                             orientation='vertical')

    view = View(HSplit(Group(VGroup(Item('tree',
                                         editor=tree_editor,
                                         show_label=False),
                                    Item('plot', show_label=False)),
                             Item('time', label='t/T_max')),
                       Item('figure', editor=MPLFigureEditor(),
                            dock='vertical', width=0.7, height=0.9),
                       show_labels=False),
                resizable=True,
                height=0.9, width=1.0
                )
if __name__ == '__main__':

    ts = TStepper()
    n_dofs = ts.domain.n_dofs
    ts.bc_list = [BCDof(var='u', dof=0, value=0.0),
                  BCDof(var='u', dof=n_dofs - 1, value=10.0)]
    tl = TLoop(ts=ts)

    tree = TreeStructure()

    window = MainWindow(tree=tree,
                        mats_eval=ts.mats_eval,
                        fets_eval=ts.fets_eval,
                        time_stepper=ts,
                        time_loop=tl)
#     window.draw()
#
    window.configure_traits()
