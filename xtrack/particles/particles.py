import numpy as np
import xobjects as xo

from ._pyparticles import Pyparticles

from ..dress import dress

from scipy.constants import m_p
from scipy.constants import e as qe
from scipy.constants import c as clight

pmass = m_p * clight * clight / qe




LAST_INVALID_STATE = -999999999

size_vars = (
    (xo.Int64,   '_capacity'),
    (xo.Int64,   '_num_active_particles'),
    (xo.Int64,   '_num_lost_particles'),
    )
# Capacity is always kept up to data
# the other two are placeholders to be used if needed
# i.e. on ContextCpu

scalar_vars = (
    (xo.Float64, 'q0'),
    (xo.Float64, 'mass0'),
    )

per_particle_vars = (
    (xo.Float64, 'p0c'),
    (xo.Float64, 'gamma0'),
    (xo.Float64, 'beta0'),
    (xo.Float64, 's'),
    (xo.Float64, 'x'),
    (xo.Float64, 'y'),
    (xo.Float64, 'px'),
    (xo.Float64, 'py'),
    (xo.Float64, 'zeta'),
    (xo.Float64, 'psigma'),
    (xo.Float64, 'delta'),
    (xo.Float64, 'rpp'),
    (xo.Float64, 'rvv'),
    (xo.Float64, 'chi'),
    (xo.Float64, 'charge_ratio'),
    (xo.Float64, 'weight'),
    (xo.Int64, 'particle_id'),
    (xo.Int64, 'at_element'),
    (xo.Int64, 'at_turn'),
    (xo.Int64, 'state'),
    (xo.Int64, 'parent_particle_id'),
    )

fields = {}
for tt, nn in size_vars + scalar_vars:
    fields[nn] = tt

for tt, nn in per_particle_vars:
    fields[nn] = tt[:]

ParticlesData = type(
        'ParticlesData',
        (xo.Struct,),
        fields)

pysixtrack_naming=(
        ('qratio', 'charge_ratio'),
        ('mratio', 'mass_ratio'),
        ('partid', 'particle_id'),
        ('turn', 'at_turn'),
        ('elemid', 'at_element'),
        )


class Particles(dress(ParticlesData)):
    _structure = {
            'size_vars': size_vars,
            'scalar_vars': scalar_vars,
            'per_particle_vars': per_particle_vars}

    def __init__(self, **kwargs):

        # Compatibility with old pysixtrack naming
        for old, new in pysixtrack_naming:
            if old in kwargs.keys():
                assert new not in kwargs.keys()
                kwargs[new] = kwargs[old]

        if '_xobject' in kwargs.keys():
            # Initialize xobject
            self.xoinitialize(**kwargs)
        else:

            if any([nn in kwargs.keys() for tt, nn in per_particle_vars]):
                # Needed to generate consistent longitudinal variables
                pyparticles = Pyparticles(**kwargs)

                part_dict = _pyparticles_to_xtrack_dict(pyparticles)
                if '_capacity' in kwargs.keys():
                    assert kwargs['_capacity'] >= part_dict['_capacity']

                else:
                    kwargs['_capacity'] = part_dict['_capacity']
            else:
                assert '_capacity' in kwargs.keys()
                pyparticles = None

            # Make sure _capacity is integer
            kwargs['_capacity'] = int(kwargs['_capacity'])

            # We just provide array sizes to xoinitialize (we will set values later)
            kwargs.update(
                    {kk: kwargs['_capacity'] for tt, kk in per_particle_vars})

            # Initialize xobject
            self.xoinitialize(**kwargs)

            # Initialize coordinates
            if pyparticles is not None:
                context = self._buffer.context
                for tt, kk in list(scalar_vars):
                    setattr(self, kk, part_dict[kk])
                for tt, kk in list(per_particle_vars):
                    vv = getattr(self, kk)
                    vals =  context.nparray_to_context_array(part_dict[kk])
                    ll = len(vals)
                    vv[:ll] = vals
                    vv[ll:] = LAST_INVALID_STATE
            else:
                for tt, kk in list(scalar_vars):
                    setattr(self, kk, 0.)

                for tt, kk in list(per_particle_vars):
                    if kk == 'chi' or kk == 'charge_ratio' or kk == 'state':
                        value = 1.
                    elif kk == 'particle_id':
                        value = np.arange(0, self._capacity, dtype=np.int64)
                    else:
                        value = 0.
                    getattr(self, kk)[:] = value

        self._num_active_particles = -1 # To be filled in only on CPU
        self._num_lost_particles = -1 # To be filled in only on CPU

        if isinstance(self._buffer.context, xo.ContextCpu):
            # Particles always need to be organized to run on CPU
            self.reorganize()

    def reorganize(self):
        assert not isinstance(self._buffer.context, xo.ContextPyopencl), (
                'Masking does not work with pyopencl')
        mask_active = self.state > 0
        mask_lost = (self.state < 1) & (self.state>LAST_INVALID_STATE)

        n_active = np.sum(mask_active)
        n_lost = np.sum(mask_lost)

        for tt, nn in self._structure['per_particle_vars']:
            vv = getattr(self, nn)
            vv_active = vv[mask_active]
            vv_lost = vv[mask_lost]

            vv[:n_active] = vv_active
            vv[n_active:n_active+n_lost] = vv_lost
            vv[n_active+n_lost:] = LAST_INVALID_STATE

        if isinstance(self._buffer.context, xo.ContextCpu):
            self._num_active_particles = n_active
            self._num_lost_particles = n_lost

        return n_active, n_lost

    def add_particles(self, part, keep_lost=False):

        if keep_lost:
            raise NotImplementedError
        assert not isinstance(self._buffer.context, xo.ContextPyopencl), (
                'Masking does not work with pyopencl')

        mask_copy = part.state > 0
        n_copy = np.sum(mask_copy)

        n_active, n_lost = self.reorganize()
        i_start_copy = n_active + n_lost
        n_free = self._capacity - n_active - n_lost

        max_id = np.max(self.particle_id[:n_active+n_lost])

        if n_copy > n_free:
            raise NotImplementedError("Out of space, need to regenerate xobject")

        for tt, nn in self._structure['scalar_vars']:
            assert np.isclose(getattr(self, nn), getattr(part, nn),
                    rtol=1e-14, atol=1e-14)

        for tt, nn in self._structure['per_particle_vars']:
            vv = getattr(self, nn)
            vv_copy = getattr(part, nn)[mask_copy]
            vv[i_start_copy:i_start_copy+n_copy] = vv_copy

        self.particle_id[i_start_copy:i_start_copy+n_copy] = np.arange(
                                     max_id+1, max_id+1+n_copy, dtype=np.int64)

        self.reorganize()

    def get_active_particle_id_range(self):
        ctx2np = self._buffer.context.nparray_from_context_array
        mask_active = ctx2np(self.state) > 0
        ids_active_particles = ctx2np(self.particle_id)[mask_active]
        # Behaves as python rante (+1)
        return np.min(ids_active_particles), np.max(ids_active_particles)+1

    def _set_p0c(self):
        energy0 = np.sqrt(self.p0c ** 2 + self.mass0 ** 2)
        self.beta0 = self.p0c / energy0
        self.gamma0 = energy0 / self.mass0

    @property
    def ptau(self):
        return (
            np.sqrt(self.delta ** 2 + 2 * self.delta + 1 / self.beta0 ** 2)
            - 1 / self.beta0
        )

    def set_reference(self, p0c=7e12, mass0=pmass, q0=1):
        self.q0 = q0
        self.mass0 = mass0
        self.p0c = p0c
        return self

    def set_particle(self, index, set_scalar_vars=False,
            check_scalar_vars=True, **kwargs):

        # Compatibility with old pysixtrack naming
        for old, new in pysixtrack_naming:
            if old in kwargs.keys():
                assert new not in kwargs.keys()
                kwargs[new] = kwargs[old]


        # Needed to generate consistent longitudinal variables
        pyparticles = Pyparticles(**kwargs)
        part_dict = pyparticles_to_xtrack_dict(pyparticles)
        for tt, kk in list(scalar_vars):
            setattr(self, kk, part_dict[kk])
        for tt, kk in list(per_particle_vars):
            getattr(self, kk)[index] = part_dict[kk][0]

    def _update_delta(self, new_delta_value):
        beta0 = self.beta0
        delta_beta0 = new_delta_value * beta0
        ptau_beta0  = np.sqrt( delta_beta0 * delta_beta0 +
                                2. * delta_beta0 * beta0 + 1. ) - 1.
        one_plus_delta = 1. + new_delta_value
        rvv    = ( one_plus_delta ) / ( 1. + ptau_beta0 )
        rpp    = 1. / one_plus_delta
        psigma = ptau_beta0 / ( beta0 * beta0 )

        self.delta[:] = new_delta_value
        self.rvv[:] = rvv
        self.rpp[:] = rpp
        self.psigma[:] = psigma


def gen_local_particle_api(mode='no_local_copy'):

    if mode != 'no_local_copy':
        raise NotImplementedError

    src_lines = []
    src_lines.append('''typedef struct{''')
    for tt, vv in size_vars + scalar_vars:
        src_lines.append('                 ' + tt._c_type + '  '+vv+';')
    for tt, vv in per_particle_vars:
        src_lines.append('    /*gpuglmem*/ ' + tt._c_type + '* '+vv+';')
    src_lines.append(    '                 int64_t ipart;')
    src_lines.append('}LocalParticle;')
    src_typedef = '\n'.join(src_lines)

    src_lines = []
    src_lines.append('''
    /*gpufun*/
    void Particles_to_LocalParticle(ParticlesData source,
                                    LocalParticle* dest,
                                    int64_t id){''')
    for tt, vv in size_vars + scalar_vars:
        src_lines.append(
                f'  dest->{vv} = ParticlesData_get_'+vv+'(source);')
    for tt, vv in per_particle_vars:
        src_lines.append(
                f'  dest->{vv} = ParticlesData_getp1_'+vv+'(source, 0);')
    src_lines.append('  dest->ipart = id;')
    src_lines.append('}')
    src_particles_to_local = '\n'.join(src_lines)

    src_lines = []
    src_lines.append('''
    /*gpufun*/
    void LocalParticle_to_Particles(
                                    LocalParticle* source,
                                    ParticlesData dest,
                                    int64_t id,
                                    int64_t set_scalar){''')
    src_lines.append('if (set_scalar){')
    for tt, vv in size_vars + scalar_vars:
        src_lines.append(
                f'  ParticlesData_set_' + vv + '(dest,'
                f'      LocalParticle_get_{vv}(source));')
    src_lines.append('}')

    for tt, vv in per_particle_vars:
        src_lines.append(
                f'  ParticlesData_set_' + vv + '(dest, id, '
                f'      LocalParticle_get_{vv}(source));')
    src_lines.append('}')
    src_local_to_particles = '\n'.join(src_lines)

    src_lines=[]
    for tt, vv in per_particle_vars:
        src_lines.append('''
    /*gpufun*/
    void LocalParticle_add_to_'''+vv+f'(LocalParticle* part, {tt._c_type} value)'
    +'{')
        src_lines.append(f'  part->{vv}[part->ipart] += value;')
        src_lines.append('}\n')
    src_adders = '\n'.join(src_lines)

    src_lines=[]
    for tt, vv in per_particle_vars:
        src_lines.append('''
    /*gpufun*/
    void LocalParticle_scale_'''+vv+f'(LocalParticle* part, {tt._c_type} value)'
    +'{')
        src_lines.append(f'  part->{vv}[part->ipart] *= value;')
        src_lines.append('}\n')
    src_scalers = '\n'.join(src_lines)

    src_lines=[]
    for tt, vv in per_particle_vars:
        src_lines.append('''
    /*gpufun*/
    void LocalParticle_set_'''+vv+f'(LocalParticle* part, {tt._c_type} value)'
    +'{')
        src_lines.append(f'  part->{vv}[part->ipart] = value;')
        src_lines.append('}')
    src_setters = '\n'.join(src_lines)

    src_lines=[]
    for tt, vv in size_vars + scalar_vars:
        src_lines.append('/*gpufun*/')
        src_lines.append(f'{tt._c_type} LocalParticle_get_'+vv
                        + f'(LocalParticle* part)'
                        + '{')
        src_lines.append(f'  return part->{vv};')
        src_lines.append('}')
    for tt, vv in per_particle_vars:
        src_lines.append('/*gpufun*/')
        src_lines.append(f'{tt._c_type} LocalParticle_get_'+vv
                        + f'(LocalParticle* part)'
                        + '{')
        src_lines.append(f'  return part->{vv}[part->ipart];')
        src_lines.append('}')
    src_getters = '\n'.join(src_lines)

    src_exchange = '''
/*gpufun*/
void LocalParticle_exchange(LocalParticle* part, int64_t i1, int64_t i2){
'''
    for tt, vv in per_particle_vars:
        src_exchange += '\n'.join([
          '\n    {',
          f'    {tt._c_type} temp = part->{vv}[i2];',
          f'    part->{vv}[i2] = part->{vv}[i1];',
          f'    part->{vv}[i1] = temp;',
          '     }'])
    src_exchange += '}\n'


    custom_source='''
/*gpufun*/
double LocalParticle_get_energy0(LocalParticle* part){

    double const p0c = LocalParticle_get_p0c(part);
    double const m0  = LocalParticle_get_mass0(part);

    return sqrt( p0c * p0c + m0 * m0 );
}

/*gpufun*/
void LocalParticle_add_to_energy(LocalParticle* part, double delta_energy){

    double const beta0 = LocalParticle_get_beta0(part);
    double const delta_beta0 = LocalParticle_get_delta(part) * beta0;

    double const ptau_beta0 =
        delta_energy / LocalParticle_get_energy0(part) +
        sqrt( delta_beta0 * delta_beta0 + 2.0 * delta_beta0 * beta0
                + 1. ) - 1.;

    double const ptau   = ptau_beta0 / beta0;
    double const psigma = ptau / beta0;
    double const delta = sqrt( ptau * ptau + 2. * psigma + 1 ) - 1;

    double const one_plus_delta = delta + 1.;
    double const rvv = one_plus_delta / ( 1. + ptau_beta0 );

    LocalParticle_set_delta(part, delta );
    LocalParticle_set_psigma(part, psigma );
    LocalParticle_scale_zeta(part,
        rvv / LocalParticle_get_rvv(part));

    LocalParticle_set_rvv(part, rvv );
    LocalParticle_set_rpp(part, 1. / one_plus_delta );
}

/*gpufun*/
double LocalParticle_get_ptau(LocalParticle* part){
    double const delta = LocalParticle_get_delta(part);
    double const beta0 = LocalParticle_get_beta0(part);
    return sqrt(delta*delta + 2 * delta + 1 / (beta0*beta0))- 1 / beta0;
}

/*gpufun*/
double LocalParticle_get_beta(LocalParticle* part){
    return (1 + LocalParticle_get_delta(part)) / (1 / LocalParticle_get_beta0(part) + LocalParticle_get_ptau(part));
}

/*gpufun*/
double LocalParticle_get_sigma(LocalParticle* part){
    double const beta0 = LocalParticle_get_beta0(part);
    double const beta = LocalParticle_get_beta(part);
    return beta0 / beta * LocalParticle_get_zeta(part);
}

/*gpufun*/
void LocalParticle_update_sigma(LocalParticle* part, double new_sigma_value){
    double const beta0 = LocalParticle_get_beta0(part);
    double const beta = LocalParticle_get_beta(part);
    double const new_zeta = beta / beta0 * new_sigma_value;
    LocalParticle_set_zeta(part, new_zeta );
}


/*gpufun*/
void LocalParticle_update_delta(LocalParticle* part, double new_delta_value){
    double const beta0 = LocalParticle_get_beta0(part);
    double const delta_beta0 = new_delta_value * beta0;
    double const ptau_beta0  = sqrt( delta_beta0 * delta_beta0 +
                                2. * delta_beta0 * beta0 + 1. ) - 1.;

    double const one_plus_delta = 1. + new_delta_value;
    double const rvv    = ( one_plus_delta ) / ( 1. + ptau_beta0 );
    double const rpp    = 1. / one_plus_delta;
    double const psigma = ptau_beta0 / ( beta0 * beta0 );

    LocalParticle_set_delta(part, new_delta_value);

    LocalParticle_set_rvv(part, rvv );
    LocalParticle_set_rpp(part, rpp );
    LocalParticle_set_psigma(part, psigma );

}

/*gpufun*/
void LocalParticle_update_psigma(LocalParticle* part, double new_psigma_value){
    double const beta0 = LocalParticle_get_beta0(part);
    double const new_ptau_value = new_psigma_value * beta0;
    double const new_delta_value = sqrt(new_ptau_value * new_ptau_value + 2 * new_ptau_value / beta0 + 1) - 1;

    double const one_plus_delta = 1. + new_delta_value;
    double const rvv    = ( one_plus_delta ) / ( 1. + new_ptau_value*beta0 );
    double const rpp    = 1. / one_plus_delta;

    LocalParticle_set_delta(part, new_delta_value);

    LocalParticle_scale_zeta(part,
        rvv / LocalParticle_get_rvv(part));

    LocalParticle_set_rvv(part, rvv );
    LocalParticle_set_rpp(part, rpp );
    LocalParticle_set_psigma(part, new_psigma_value );
}

/*gpufun*/
void LocalParticle_update_p0c(LocalParticle* part, double new_p0c_value){

    double const mass0 = LocalParticle_get_mass0(part);
    double const old_p0c = LocalParticle_get_p0c(part);
    double const old_delta = LocalParticle_get_delta(part);

    double const ppc = old_p0c * old_delta + old_p0c;
    double const new_delta = (ppc - new_p0c_value)/new_p0c_value;

    double const new_energy0 = sqrt(new_p0c_value*new_p0c_value + mass0 * mass0);
    double const new_beta0 = new_p0c_value / new_energy0;
    double const new_gamma0 = new_energy0 / mass0;

    LocalParticle_set_p0c(part, new_p0c_value);
    LocalParticle_set_gamma0(part, new_gamma0);
    LocalParticle_set_beta0(part, new_beta0);

    LocalParticle_update_delta(part, new_delta);

    LocalParticle_scale_px(part, old_p0c/new_p0c_value);
    LocalParticle_scale_py(part, old_p0c/new_p0c_value);

}
'''

    source = '\n\n'.join([src_typedef, src_adders, src_getters,
                          src_setters, src_scalers, src_exchange,
                          src_particles_to_local, src_local_to_particles,
                          custom_source])

    return source

def pyparticles_to_xtrack_dict(pyparticles):

    out = {}

    pyst_dict = pyparticles.to_dict()
    for old, new in pysixtrack_naming:
        if hasattr(pyparticles, old):
            assert new not in pyst_dict.keys()
            pyst_dict[new] = getattr(pyparticles, old)

    if hasattr(pyparticles, 'weight'):
        pyst_dict['weight'] = getattr(pyparticles, 'weight')
    else:
        pyst_dict['weight'] = 1.

    for tt, kk in scalar_vars + per_particle_vars:
        # Use properties
        pyst_dict[kk] = getattr(pyparticles, kk)


    for kk, vv in pyst_dict.items():
        pyst_dict[kk] = np.atleast_1d(vv)

    lll = [len(vv) for kk, vv in pyst_dict.items() if hasattr(vv, '__len__')]
    lll = list(set(lll))
    assert len(set(lll) - {1}) <= 1
    _capacity = max(lll)
    out['_capacity'] = _capacity

    for tt, kk in scalar_vars:
        val = pyst_dict[kk]
        assert np.allclose(val, val[0], rtol=1e-10, atol=1e-14)
        out[kk] = val[0]

    for tt, kk in per_particle_vars:

        val_pyst = pyst_dict[kk]

        if _capacity > 1 and len(val_pyst)==1:
            temp = np.zeros(int(_capacity), dtype=tt._dtype)
            temp += val_pyst[0]
            val_pyst = temp

        if type(val_pyst) != tt._dtype:
            val_pyst = np.array(val_pyst, dtype=tt._dtype)

        out[kk] = val_pyst

    #out['_num_active_particles'] = np.sum(out['state']>0)
    #out['_num_lost_particles'] = np.sum((out['state'] < 0) &
    #                                      (out['state'] > LAST_INVALID_STATE))

    return out
