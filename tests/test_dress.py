import numpy as np
from xtrack import dress
import xobjects as xo

def test_dress():
    class ElementData(xo.Struct):
        n = xo.Int32
        b = xo.Float64
        vv = xo.Float64[:]

    class Element(dress(ElementData)):

        def __init__(self, vv, **kwargs):
            self.xoinitialize(n=len(vv), b=np.sum(vv), vv=vv,
                              **kwargs)
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        ele = Element([1,2,3], _context=context)
        assert ele.n == ele._xobject.n == 3
        assert ele.b == ele._xobject.b == 6
        assert ele.vv[1] == ele._xobject.vv[1] == 2

        new_vv = context.nparray_to_context_array(np.array([7,8,9]))
        ele.vv = new_vv
        assert ele.n == ele._xobject.n == 3
        assert ele.b == ele._xobject.b == 6
        assert ele.vv[1] == ele._xobject.vv[1] == 8

        ele.n = 5.
        assert ele.n == ele._xobject.n == 5

        ele.b = 50
        assert ele.b == ele._xobject.b == 50.


def test_explicit_buffer():
    class ElementData(xo.Struct):
        n = xo.Int32
        b = xo.Float64
        vv = xo.Float64[:]

    class Element(dress(ElementData)):

        def __init__(self, vv, **kwargs):
            self.xoinitialize(n=len(vv), b=np.sum(vv), vv=vv, **kwargs)

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")
        ele1 = Element([1,2,3], _context=context)
        ele2 = Element([7,8,9], _buffer=ele1._buffer)

        assert ele1.vv[1] == ele1._xobject.vv[1] == 2
        assert ele2.vv[1] == ele2._xobject.vv[1] == 8
        for ee in [ele1, ele2]:
            assert (ee._buffer is ee._xobject._buffer)
            assert (ee._offset == ee._xobject._offset)

        assert ele1._buffer is ele2._buffer
        assert ele1._offset != ele2._offset
