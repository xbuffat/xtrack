import time

import numpy as np
from matplotlib import pyplot as plt
from scipy import constants

import xobjects as xo
context = xo.ContextCpu(omp_num_threads=0)
import xtrack as xt
import xfields as xf

muonMass = constants.value('muon mass energy equivalent in MeV')*1E6

n_macroparticles = 10
energy = 1.25 # [GeV]
p0c = np.sqrt((energy*1E9)**2-muonMass**2)
gamma = np.sqrt(1+(p0c/muonMass)**2)
beta = np.sqrt(1-1/gamma**2)
normemit_x = 25E-6
normemit_y = 25E-6
sigma_z = 4.6E-2
emit_z = 7.5E-3
sigma_delta = emit_z/sigma_z/energy
beta_s = sigma_z/sigma_delta # need to convert to sigma/psigma?

beta_x = 10.0
beta_y = 10.0
alpha_x = 0.0
alpha_y = 0.0
dispersion = 0.0

nTurn = 10000


particles = xt.Particles(_context=context,
                         q0 = 1,
                         mass0 = muonMass,
                         p0c=p0c,
                         x=np.sqrt(normemit_x*beta_x/gamma)*np.linspace(0,6,n_macroparticles),
                         px=np.sqrt(normemit_x/beta/gamma)*np.zeros(n_macroparticles,dtype=float),
                         y=np.sqrt(normemit_y*beta_y/gamma)*np.linspace(0,6,n_macroparticles),
                         py=np.sqrt(normemit_y/beta_y/gamma)*np.zeros(n_macroparticles,dtype=float),
                         zeta=sigma_z*np.zeros(n_macroparticles),
                         delta=sigma_delta*np.ones(n_macroparticles,dtype=float),
                         )

arc = xt.LinearTransferMatrix(alpha_x_0 = alpha_x, beta_x_0 = beta_x, disp_x_0 = dispersion,
                           alpha_x_1 = alpha_x, beta_x_1 = beta_x, disp_x_1 = dispersion,
                           alpha_y_0 = alpha_y, beta_y_0 = beta_y, disp_y_0 = 0.0,
                           alpha_y_1 = alpha_y, beta_y_1 = beta_y, disp_y_1 = 0.0,
                           Q_x = 0.155, Q_y=0.16,
                           beta_s = beta_s, Q_s = 0.155,
                           energy_ref_increment=0.0,energy_increment=0.0)


for turn in range(nTurn):
    arc.track(particles)
    plt.figure(0)
    plt.plot(particles.zeta,particles.delta,'.r')

particles = xt.Particles(_context=context,
                         q0 = 1,
                         mass0 = muonMass,
                         p0c=p0c,
                         x=np.sqrt(normemit_x*beta_x/gamma)*np.linspace(0,6,n_macroparticles),
                         px=np.sqrt(normemit_x/beta/gamma)*np.zeros(n_macroparticles,dtype=float),
                         y=np.sqrt(normemit_y*beta_y/gamma)*np.linspace(0,6,n_macroparticles),
                         py=np.sqrt(normemit_y/beta_y/gamma)*np.zeros(n_macroparticles,dtype=float),
                         zeta=sigma_z*np.zeros(n_macroparticles),
                         delta=sigma_delta*np.ones(n_macroparticles,dtype=float),
                         )

arc = xt.LinearTransferMatrixWithSigma(alpha_x_0 = alpha_x, beta_x_0 = beta_x, disp_x_0 = dispersion,
                           alpha_x_1 = alpha_x, beta_x_1 = beta_x, disp_x_1 = dispersion,
                           alpha_y_0 = alpha_y, beta_y_0 = beta_y, disp_y_0 = 0.0,
                           alpha_y_1 = alpha_y, beta_y_1 = beta_y, disp_y_1 = 0.0,
                           Q_x = 0.155, Q_y=0.16,
                           beta_s = beta_s, Q_s = 0.155,
                           energy_ref_increment=0.0,energy_increment=0.0)

for turn in range(nTurn):
    arc.track(particles)
    plt.figure(0)
    plt.plot(particles.zeta,particles.delta,'.b')

plt.figure(0)
plt.xlabel(r'$\zeta$')
plt.ylabel(r'$\delta$')

plt.show()



