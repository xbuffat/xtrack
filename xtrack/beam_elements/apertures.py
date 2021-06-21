import xobjects as xo

from ..dress_element import BeamElement
from ..general import _pkg_root

class LimitRect(BeamElement):
    _xofields={
        'min_x': xo.Float64,
        'max_x': xo.Float64,
        'min_y': xo.Float64,
        'max_y': xo.Float64,
        }
LimitRect.XoStruct.extra_sources = [
        _pkg_root.joinpath('beam_elements/apertures_src/limitrect.h')]

