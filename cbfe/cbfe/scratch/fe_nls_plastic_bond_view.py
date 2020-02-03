'''
Created on 07.01.2016

@author: Yingxiong
'''
import numpy as np
from .fe_nls_plastic_bond import MATSEval, FETS1D52ULRH, TStepper, TLoop
from matplotlib import pyplot as plt
from ibvpy.api import BCDof
from traits.api import HasTraits, Property, Instance, cached_property, Str, Button, Range, on_trait_change, Array, List
from matplotlib.figure import Figure
from .mpl_figure_editor import MPLFigureEditor
from traitsui.api import View, Item, Group, HSplit, Handler, InstanceEditor, UItem, VGroup


class Mainwindow(HasTraits):

    #     panel = Instance(ControlPanel)
    mats_eval = Instance(MATSEval)

    fets_eval = Instance(FETS1D52ULRH)

    time_stepper = Instance(TStepper)

    time_loop = Instance(TLoop)

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
        self.draw()
        self.figure.canvas.draw()

    sigma_y = Range(0.2, 1.2)
    E_b = Range(0.05, 0.35)
    K_bar = Range(-0.01, 0.05)

    @on_trait_change('sigma_y, E_b, K_bar')
    def plot(self):
        self.mats_eval.sigma_y = self.sigma_y
        self.mats_eval.E_b = self.E_b
        self.mats_eval.K_bar = self.K_bar
        self.draw()
        self.figure.canvas.draw()

    L_x = Range(5., 15., value=15.)

    @on_trait_change('L_x')
    def plot1(self):
        self.time_stepper.L_x = self.L_x
        self.draw()
        self.figure.canvas.draw()

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

        slip, bond = self.time_stepper.mats_eval.get_bond_slip()
        self.ax1.cla()
        l_bs, = self.ax1.plot(slip, bond)
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

    time = Range(0.00, 1.02, value=1.02)

    @on_trait_change('time')
    def draw_t(self):
        idx = (np.abs(self.time - self.t_record)).argmin()
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

        self.figure.canvas.draw()

    view = View(HSplit(Item('figure', editor=MPLFigureEditor(),
                            dock='vertical', width=0.7, height=0.9),
                       Group(Item('mats_eval'),
                             Item('fets_eval'),
                             Item('time_stepper'),
                             Item('time_loop'),
                             Item('sigma_y'),
                             Item('E_b'),
                             Item('K_bar'),
                             Item('L_x'),
                             Item('time')),
                       show_labels=False),
                resizable=True,
                height=0.9, width=1.0,
                )

if __name__ == '__main__':

    ts = TStepper()
    n_dofs = ts.domain.n_dofs
    ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),
                  BCDof(var='u', dof=n_dofs - 1, value=5.0)]
    tl = TLoop(ts=ts)

    window = Mainwindow(mats_eval=ts.mats_eval,
                        fets_eval=ts.fets_eval,
                        time_stepper=ts,
                        time_loop=tl)
    window.draw()

    window.configure_traits()
