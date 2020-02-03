

from math import sin

from ibvpy.core.i_sdomain import \
    ISDomain
from ibvpy.core.sdomain import \
    SDomain
from ibvpy.mesh.cell_grid.cell_array import CellView, ICellView, CellArray, ICellArraySource
from ibvpy.mesh.cell_grid.cell_spec import CellSpec, GridCell
from ibvpy.plugins.mayavi_util.pipelines import \
    MVPolyData, MVPointLabels, MVStructuredGrid
from numpy import \
    array, unique, min, max, mgrid, ogrid, c_, alltrue, repeat, ix_, \
    arange, ones, zeros, multiply, sort, index_exp, indices, add, hstack, \
    frompyfunc, where
from traits.api import \
    HasTraits, List, Array, Property, cached_property, \
    Instance, Trait, Button, on_trait_change, Tuple, \
    Int, Float, provides, Delegate, Interface
from traitsui.api import \
    View, Item


class ILevelSetFn(Interface):
    def level_set_fn(self, x, y):
        '''Level set function evaluation.
        '''
        raise NotImplementedError


@provides(ILevelSetFn)
class SinLSF(HasTraits):
    a = Float(1.5, enter_set=True, auto_set=False)
    b = Float(2.0, enter_set=True, auto_set=False)

    def level_set_fn(self, x, y):
        '''Level set function evaluation.
        '''
        return y - (sin(self.b * x) + self.a)


@provides(ILevelSetFn)
class PlaneLSF(HasTraits):
    a = Float(.5, enter_set=True, auto_set=False)
    b = Float(2.0, enter_set=True, auto_set=False)
    c = Float(-2.5, enter_set=True, auto_set=False)

    def level_set_fn(self, x, y):
        '''Level set function evaluation.
        '''
        return self.a * x + self.b * y + self.c


@provides(ILevelSetFn)
class ElipseLSF(HasTraits):
    a = Float(.5, enter_set=True, auto_set=False)
    b = Float(2.0, enter_set=True, auto_set=False)
    c = Float(-2.5, enter_set=True, auto_set=False)

    def level_set_fn(self, x, y):
        '''Level set function evaluation.
        '''
        return self.a * x * x + self.b * y * y - self.c
