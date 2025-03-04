import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

def test_rect_ellipse():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        np2ctx = context.nparray_to_context_array
        ctx2np = context.nparray_from_context_array

        aper_rect_ellipse = xt.LimitRectEllipse(_context=context,
                max_x=23e-3, max_y=18e-3, a=23e-2, b=23e-2)
        aper_ellipse = xt.LimitRectEllipse(_context=context,
                                           a=23e-2, b=23e-2)
        aper_rect = xt.LimitRect(_context=context,
                                 max_x=23e-3, min_x=-23e-3,
                                 max_y=18e-3, min_y=-18e-3)

        XX, YY = np.meshgrid(np.linspace(-30e-3, 30e-3, 100),
                             np.linspace(-30e-3, 30e-3, 100))
        x_part = XX.flatten()
        y_part = XX.flatten()
        part_re = xp.Particles(_context=context,
                               x=x_part, y=y_part)
        part_e = part_re.copy()
        part_r = part_re.copy()

        aper_rect_ellipse.track(part_re)
        aper_ellipse.track(part_e)
        aper_rect.track(part_r)

        flag_re = ctx2np(part_re.state)[np.argsort(ctx2np(part_re.particle_id))]
        flag_r = ctx2np(part_r.state)[np.argsort(ctx2np(part_r.particle_id))]
        flag_e = ctx2np(part_e.state)[np.argsort(ctx2np(part_e.particle_id))]

        assert np.all(flag_re == (flag_r & flag_e))

def test_aperture_racetrack():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        part_gen_range = 0.11
        n_part=100000

        aper = xt.LimitRacetrack(_context=context,
                                 min_x=-5e-2, max_x=10e-2,
                                 min_y=-2e-2, max_y=4e-2,
                                 a=2e-2, b=1e-2)

        xy_out = np.array([
            [-4.8e-2, 3.7e-2],
            [9.6e-2, 3.7e-2],
            [-4.5e-2, -1.8e-2],
            [9.8e-2, -1.8e-2],
            ])

        xy_in = np.array([
            [-4.2e-2, 3.3e-2],
            [9.4e-2, 3.6e-2],
            [-3.8e-2, -1.8e-2],
            [9.2e-2, -1.8e-2],
            ])

        xy_all = np.concatenate([xy_out, xy_in], axis=0)

        particles = xp.Particles(_context=context,
                p0c=6500e9,
                x=xy_all[:, 0],
                y=xy_all[:, 1])

        aper.track(particles)

        part_state = context.nparray_from_context_array(particles.state)
        part_id = context.nparray_from_context_array(particles.particle_id)

        assert np.all(part_state[part_id<4] == 0)
        assert np.all(part_state[part_id>=4] == 1)



def test_aperture_polygon():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        np2ctx = context.nparray_to_context_array
        ctx2np = context.nparray_from_context_array

        x_vertices=np.array([1.5, 0.2, -1, -1,  1])*1e-2
        y_vertices=np.array([1.3, 0.5,  1, -1, -1])*1e-2

        aper = xt.LimitPolygon(
                        _context=context,
                        x_vertices=np2ctx(x_vertices),
                        y_vertices=np2ctx(y_vertices))

        # Try some particles inside
        parttest = xp.Particles(
                        _context=context,
                        p0c=6500e9,
                        x=x_vertices*0.99,
                        y=y_vertices*0.99)
        aper.track(parttest)
        assert np.allclose(ctx2np(parttest.state), 1)

        # Try some particles outside
        parttest = xp.Particles(
                        _context=context,
                        p0c=6500e9,
                        x=x_vertices*1.01,
                        y=y_vertices*1.01)
        aper.track(parttest)
        assert np.allclose(ctx2np(parttest.state), 0)

def test_mad_import():

    from cpymad.madx import Madx

    mad = Madx()

    mad.input("""
        m_circle: marker, apertype="circle", aperture={.2};
        m_ellipse: marker, apertype="ellipse", aperture={.2, .1};
        m_rectangle: marker, apertype="rectangle", aperture={.07, .05};
        m_rectellipse: marker, apertype="rectellipse", aperture={.2, .4, .25, .45};
        m_racetrack: marker, apertype="racetrack", aperture={.6,.4,.2,.1};
        m_octagon: marker, apertype="octagon", aperture={.4, .5, 0.5, 1.};
        beam;
        ss: sequence,l=1;
            m_circle, at=0;
            m_ellipse, at=0.01;
            m_rectangle, at=0.02;
            m_rectellipse, at=0.03;
            m_racetrack, at=0.04;
            m_octagon, at=0.05;
        endsequence;

        use,sequence=ss;
        twiss,betx=1,bety=1;
        """
        )

    line = xt.Line.from_madx_sequence(mad.sequence.ss, install_apertures=True)

    apertures = [ee for ee in line.elements if ee.__class__.__name__.startswith('Limit')]

    circ = apertures[0]
    assert circ.__class__.__name__ == 'LimitEllipse'
    assert np.isclose(circ.a_squ, .2**2, atol=1e-13, rtol=0)
    assert np.isclose(circ.b_squ, .2**2, atol=1e-13, rtol=0)

    ellip = apertures[1]
    assert ellip.__class__.__name__ == 'LimitEllipse'
    assert np.isclose(ellip.a_squ, .2**2, atol=1e-13, rtol=0)
    assert np.isclose(ellip.b_squ, .1**2, atol=1e-13, rtol=0)

    rect = apertures[2]
    assert rect.__class__.__name__ == 'LimitRect'
    assert rect.min_x == -.07
    assert rect.max_x == +.07
    assert rect.min_y == -.05
    assert rect.max_y == +.05

    rectellip = apertures[3]
    assert rectellip.max_x == .2
    assert rectellip.max_y == .4
    assert np.isclose(rectellip.a_squ, .25**2, atol=1e-13, rtol=0)
    assert np.isclose(rectellip.b_squ, .45**2, atol=1e-13, rtol=0)

    racetr = apertures[4]
    assert racetr.__class__.__name__ == 'LimitRacetrack'
    assert racetr.min_x == -.6
    assert racetr.max_x == +.6
    assert racetr.min_y == -.4
    assert racetr.max_y == +.4
    assert racetr.a == .2
    assert racetr.b == .1

    octag = apertures[5]
    assert octag.__class__.__name__ == 'LimitPolygon'
    assert octag._xobject.x_vertices[0] == 0.4
    assert np.isclose(octag._xobject.y_vertices[0], 0.4*np.tan(0.5), atol=1e-14, rtol=0)
    assert octag._xobject.y_vertices[1] == 0.5
    assert np.isclose(octag._xobject.x_vertices[1], 0.5/np.tan(1.), atol=1e-14, rtol=0)

    assert octag._xobject.y_vertices[2] == 0.5
    assert np.isclose(octag._xobject.x_vertices[2], -0.5/np.tan(1.), atol=1e-14, rtol=0)
    assert octag._xobject.x_vertices[3] == -0.4
    assert np.isclose(octag._xobject.y_vertices[3], 0.4*np.tan(0.5), atol=1e-14, rtol=0)


    assert octag._xobject.x_vertices[4] == -0.4
    assert np.isclose(octag._xobject.y_vertices[4], -0.4*np.tan(0.5), atol=1e-14, rtol=0)
    assert octag._xobject.y_vertices[5] == -0.5
    assert np.isclose(octag._xobject.x_vertices[5], -0.5/np.tan(1.), atol=1e-14, rtol=0)


    assert octag._xobject.y_vertices[6] == -0.5
    assert np.isclose(octag._xobject.x_vertices[6], 0.5/np.tan(1.), atol=1e-14, rtol=0)
    assert octag._xobject.x_vertices[7] == 0.4
    assert np.isclose(octag._xobject.y_vertices[7], -0.4*np.tan(0.5), atol=1e-14, rtol=0)
