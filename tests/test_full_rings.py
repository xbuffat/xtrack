import pickle
import json
import pathlib
import numpy as np

import xtrack as xt
import xobjects as xo
import xline as xl

from xobjects.context import available

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()



def test_full_rings(element_by_element=False):
     test_fnames =  [
            test_data_folder.joinpath('lhc_no_bb/line_and_particle.json'),
            test_data_folder.joinpath('hllhc_14/line_and_particle.json'),
            test_data_folder.joinpath('sps_w_spacecharge/'
                                      'line_with_spacecharge_and_particle.json'),
            ]

     tolerances_10_turns = [
                    (1e-9, 1e-11),
                    (1e-9, 1e-11),
                    (2e-8, 7e-9)]

     for icase, fname_line_particles in enumerate(test_fnames):

        rtol_10turns, atol_10turns = tolerances_10_turns[icase]

        print('Case:', fname_line_particles)
        for context in xo.context.get_test_contexts():
            print(f"Test {context.__class__}")

            #############
            # Load file #
            #############

            if str(fname_line_particles).endswith('.pkl'):
                with open(fname_line_particles, 'rb') as fid:
                    input_data = pickle.load(fid)
            elif str(fname_line_particles).endswith('.json'):
                with open(fname_line_particles, 'r') as fid:
                    input_data = json.load(fid)

            ##################
            # Get a sequence #
            ##################

            sequence = xl.Line.from_dict(input_data['line'])

            ##################
            # Build TrackJob #
            ##################
            print('Build tracker...')
            tracker = xt.Tracker(_context=context, sequence=sequence)

            ######################
            # Get some particles #
            ######################
            particles = xt.Particles(_context=context, **input_data['particle'])

            #########
            # Track #
            #########
            print('Track a few turns...')
            n_turns = 10
            tracker.track(particles, num_turns=n_turns)

            #######################
            # Check against xline #
            #######################
            print('Check against ...')
            ip_check = 0
            vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
            pyst_part = xl.Particles.from_dict(input_data['particle'])
            for _ in range(n_turns):
                sequence.track(pyst_part)

            for vv in vars_to_check:
                pyst_value = getattr(pyst_part, vv)
                xt_value = context.nparray_from_context_array(getattr(particles, vv))[ip_check]
                passed = np.isclose(xt_value, pyst_value,
                                    rtol=rtol_10turns, atol=atol_10turns)
                print(f'Varable {vv}:\n'
                      f'    pyst:   {pyst_value: .7e}\n'
                      f'    xtrack: {xt_value: .7e}\n')
                if not passed:
                    raise ValueError('Discrepancy found!')

            print('Test passed!')

            ##############
            # Check  ebe #
            ##############
            if element_by_element:
                print('Check element-by-element against xline...')
                pyst_part = xl.Particles.from_dict(input_data['particle'])
                vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
                problem_found = False
                for ii, (eepyst, nn) in enumerate(zip(sequence.elements, sequence.element_names)):
                    vars_before = {vv :getattr(pyst_part, vv) for vv in vars_to_check}
                    particles.set_particle(ip_check, **pyst_part.to_dict())

                    tracker.track(particles, ele_start=ii, num_elements=1)

                    eepyst.track(pyst_part)
                    for vv in vars_to_check:
                        pyst_change = getattr(pyst_part, vv) - vars_before[vv]
                        xt_change = context.nparray_from_context_array(
                                getattr(particles, vv))[ip_check] -vars_before[vv]
                        passed = np.isclose(xt_change, pyst_change, rtol=1e-10, atol=5e-14)
                        if not passed:
                            problem_found = True
                            print(f'Not passend on var {vv}!\n'
                                  f'    pyst:   {pyst_change: .7e}\n'
                                  f'    xtrack: {xt_change: .7e}\n')
                            break

                    if not passed:
                        print(f'\nelement {nn}')
                        break
                    else:
                        print(f'Check passed for element: {nn}              ',
                                end='\r', flush=True)

                if not problem_found:
                    print('All passed on context:')
                    print(context)
