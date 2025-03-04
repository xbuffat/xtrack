import xtrack as xt
import xpart as xp
import xobjects as xo

from cpymad.madx import Madx

#########################################################################
# To import a MAD-X deferred expressions together with a MAD-X sequence #
# we proceed as follows:                                                #
#########################################################################

# Load sequence in MAD-X
mad = Madx()
mad.call('../../test_data/hllhc15_noerrors_nobb/sequence.madx')
mad.use(sequence="lhcb1")

# Build Xtrack line importing MAD-X expressions
line = xt.Line.from_madx_sequence(mad.sequence['lhcb1'],
                                  deferred_expressions=True # <--
                                  )
# Define reference particle
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
                                 gamma0=mad.sequence.lhcb1.beam.gamma)

# Build tracker
tracker = xt.Tracker(line=line)

#########################################################################
# MAD-X variables can be found in `tracker.vars` or, equivalently, in   #
# `line.vars`. They can be used to change properties in the beamline.   #
# For example, we consider the MAD-X variable `on_x1` that controls     #
# the beam angle in the interaction point 1 (IP1). It is defined in     #
# microrad units.                                                       #
#########################################################################

# Inspect the value of the variable
print(tracker.vars['on_x1']._value)
# ---> returns 1 (as defined in the import)

# Measure vertical angle at the interaction point 1 (IP1)
print(tracker.twiss(at_elements=['ip1'])['px'])
# ---> returns 1e-6

# Set crossing angle using the variable
tracker.vars['on_x1'] = 300

# Measure vertical angle at the interaction point 1 (IP1)
print(tracker.twiss(at_elements=['ip1'])['px'])
# ---> returns 0.00030035

#########################################################################
# The expressions relating the beam elements properties to the          #
# variables can be inspected and modified through the data structure    #
# `tracker.element_refs` or equivalently `line.element_refs`            #
#########################################################################

# For example we can che how the dipole corrector 'mcbyv.4r1.b1' is controlled:
print(tracker.element_refs['mcbxfah.3r1'].knl[0]._expr)
# ---> returns "(-vars['acbxh3.r1'])"

# We can see that the variable controlling the corrector is in turn controlled
# by an expression involving several other variables:
print(tracker.vars['acbxh3.r1']._expr)
# ---> returns
#         (((((((-3.529000650090648e-07*vars['on_x1hs'])
#         -(1.349958221397232e-07*vars['on_x1hl']))
#          +(1.154711348310621e-05*vars['on_sep1h']))
#          +(1.535247516521591e-05*vars['on_o1h']))
#          -(9.919546388675102e-07*vars['on_a1h']))
#          +(3.769003853335184e-05*vars['on_ccpr1h']))
#           +(1.197587664190056e-05*vars['on_ccmr1h']))

# It is possible to get the list of all entities controlled by a given
# variable by using the method `_find_dependant_targets`:
tracker.vars['on_x1']._find_dependant_targets()
# ---> returns
#         [vars['on_x1'],
#          vars['on_x1hl'],
#          vars['on_dx1hl'],
#          vars['on_x1hs'],
#          vars['acbxh3.l1'],
#          element_refs['mcbxfah.3l1'],
#          element_refs['mcbxfah.3l1'].knl[0],
#          element_refs['mcbxfah.3l1'].knl,
#            ...............

#########################################################################
# The Xtrack line including the related expressions can be saved in a   #
# json and reloaded.                                                    #
#########################################################################

# Save
import json
with open('status.json', 'w') as fid:
    json.dump(line.to_dict(), fid,
    cls=xo.JEncoder)

# Reload
with open('status.json', 'r') as fid:
    dct = json.load(fid)
line_reloaded = xt.Line.from_dict(dct)

#!end-doc-part

import numpy as np
line.vars['on_x1'] = 250
assert np.isclose(tracker.twiss(at_elements=['ip1'])['px'][0], 250e-6,
                  atol=1e-6, rtol=0)

line.vars['on_x1'] = -300
assert np.isclose(tracker.twiss(at_elements=['ip1'])['px'][0], -300e-6,
                  atol=1e-6, rtol=0)
