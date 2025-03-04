#ifndef XTRACK_RFMULTIPOLE_H
#define XTRACK_RFMULTIPOLE_H

/*gpufun*/
void RFMultipole_track_local_particle(RFMultipoleData el, LocalParticle* part0){

    /*gpuglmem*/ double const* bal = RFMultipoleData_getp1_bal(el, 0);
    /*gpuglmem*/ double const* phase = RFMultipoleData_getp1_phase(el, 0);
    int64_t const order = RFMultipoleData_get_order(el);
    double const frequency = RFMultipoleData_get_frequency(el);
    double const voltage = RFMultipoleData_get_voltage(el);
    double const lag = RFMultipoleData_get_lag(el);

    //start_per_particle_block (part0->part)
        double const k = frequency * ( 2.0 * PI / C_LIGHT);

        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);
        double const beta0  = LocalParticle_get_beta0(part);
        double const zeta   = LocalParticle_get_zeta(part);
        double const q      = LocalParticle_get_q0(part)
                            * LocalParticle_get_charge_ratio(part);
        double const rvv    = LocalParticle_get_rvv(part);
        double const ktau   = k * zeta / ( beta0 * rvv );

        double dpx = 0.0;
        double dpy = 0.0;
        double dptr = 0.0;
        double zre = 1.0;
        double zim = 0.0;

        for (int64_t kk = 0; kk <= order; kk++)
        {
            double const pn_kk = DEG2RAD * phase[2*kk] - ktau;
            double const ps_kk = DEG2RAD * phase[2*kk+1] - ktau;

            double const bal_n_kk = bal[2*kk];
            double const bal_s_kk = bal[2*kk+1];

            double const cn = cos(pn_kk);
            double const cs = cos(ps_kk);
            double const sn = sin(pn_kk);
            double const ss = sin(ps_kk);

            dpx += cn * (bal_n_kk * zre) - cs * (bal_s_kk * zim);
            dpy += cs * (bal_s_kk * zre) + cn * (bal_n_kk * zim);

            double const zret = zre * x - zim * y;
            zim = zim * x + zre * y;
            zre = zret;

            dptr += sn * (bal_n_kk * zre) - ss * (bal_s_kk * zim);
        }

        double const cav_energy = q * voltage * sin(lag * DEG2RAD - ktau);
        double const p0c = LocalParticle_get_p0c(part);
        double const rfmultipole_energy = - q * ( (k * p0c) * dptr );

        double const chi    = LocalParticle_get_chi(part);

        double const px_kick = - chi * dpx;
        double const py_kick =   chi * dpy;
        double const energy_kick = cav_energy + rfmultipole_energy;

        LocalParticle_add_to_px(part, px_kick);
        LocalParticle_add_to_py(part, py_kick);
        LocalParticle_add_to_energy(part, energy_kick, 1);

    //end_per_particle_block

}

#endif
