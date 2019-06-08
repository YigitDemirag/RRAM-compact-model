# -*- coding: utf-8 -*-
"""
 Compact Modeling for RRAM Devices
 =======================================
 This script calculates IV characteristics of a bilayer RRAM device (HOTO)
 as described in the following article:

 Hardtdegen, A. et al. Improved Switching Stability and the 
 Effect of an Internal Series Resistor in HfO2/TiOx Bilayer 
 ReRAM Cells. IEEE Transactions on Electron Devices 65, 3229–3236(2018).
 
 Author: Yigit Demirag, INI | ETHZ | UZH
 Date  : May 7, 2019
"""
import scipy.constants as const
from scipy.integrate import odeint
from scipy import optimize
import math
import matplotlib.pyplot as plt
import numpy as np

### Parameters #######################
eV_to_J     =  const.physical_constants['electron volt-joule relationship'][0]
J_to_eV     =  const.physical_constants['joule-electron volt relationship'][0]
kB          =  const.Boltzmann # J/K
h           =  const.Planck # Js
e           =  const.elementary_charge # C
mu_star     =  const.electron_mass # kg
eps_0       =  const.epsilon_0 # F/m
eps         =  17 * eps_0 # F/m
eps_phi_B   =  5.5 * eps_0 # F/m
z_vo        =  2       # 1
l_cell      =  3e-9    # m
l_disc      =  1e-9    # m
r_fil       =  30e-9   # m
a           =  0.4e-9  # m
nu_0        =  3e11    # Hz
del_WA      =  0.9     # eV
T_0         =  293     # Kelvin
A_star      =  6.01e5  # A/(m^2K^2)
e_phi_Bn0   =  0.3     # eV
e_phi_n     =  0.1     # eV
mu_n        =  1e-5    # m^2/(Vs)
N_plug      =  2e27    # m^-3
N_disc_max  =  2e27    # m^-3
N_disc_min  =  4e24    # m^-3 (HOTO)
R_series    =  1200    # Ohm (HOTO)
R_th_eff    =  1.06e6  # K/W (HOTO)
sweeprate   =  0.67    # V/s
#######################################

def coth(x):
    """Returns hyporbolic cotangent
    """
    return math.cosh(x)/math.sinh(x)

def eq_V_sch(V_sch, V_app, T, N_disc, R_disc, R_plug):
    """Kirchoff Current Law for V_sch
       0 = V_app - I_sch * (R_series + R_plug + R_disc) - V_sch
    """
    if V_app < 0: # RESET case
        e_phi_Bn = e_phi_Bn0 - e * J_to_eV * pow((e**3 * z_vo * N_disc * ((e_phi_Bn0*eV_to_J/e) - (e_phi_n*eV_to_J/e) + V_sch))/ \
            (8 * math.pi**2 * eps_phi_B**3) , 1/4) # eV -  Effective barrier height (V < 0)
       
        ### DEBUG ######
        '''
        print(V_sch)
        if ((e_phi_Bn0*eV_to_J/e) - (e_phi_n*eV_to_J/e) + V_sch) < 0:
            ic = (e**3 * z_vo * N_disc * ((e_phi_Bn0*eV_to_J/e) - (e_phi_n*eV_to_J/e) + V_sch))/ \
            (8 * math.pi**2 * eps_phi_B**3)
            print("N_disc     :  %.2e" %N_disc)
            print("Root inside:  %.2e" %ic)
            print((e_phi_Bn0*eV_to_J/e) - (e_phi_n*eV_to_J/e) + V_sch)
            print(pow(ic,1/4))
        '''
        #### DEBUG #####
       
        I_sch = -A * A_star * T**2 * math.exp(-(e_phi_Bn * eV_to_J) / (kB * T)) * \
            (math.exp(e * (-V_sch / (kB * T))) - 1) # A - Current passing device
    
    else: # SET case
        W_00 = (e * h) * math.sqrt((z_vo * N_disc)/(mu_star * eps)) / (4 * math.pi) # J
        W_0 = W_00 * coth(W_00 / (kB * T)) # J
        xi = W_00 / ((W_00/(kB * T)) - math.tanh(W_00 / (kB * T))) # J    
        
        e_phi_Bn = e_phi_Bn0 - e * J_to_eV * pow((e**3 * z_vo * N_disc * ((e_phi_Bn0*eV_to_J/e) - (e_phi_n*eV_to_J/e) + V_sch))/ \
            (8 * math.pi**2 * eps_phi_B**3) , 1/4) # eV -  Effective barrier height (V < 0)

        I_sch = A * A_star * (T/kB) * \
            math.sqrt(math.pi * W_00 * e * (V_sch + ((e_phi_Bn0 * eV_to_J / e) / (pow(math.cosh(W_00 / (kB * T)), 2) )))) * \
            math.exp((-e_phi_Bn*eV_to_J)/(W_0)) * (math.exp(e * V_sch / xi) - 1)

    return V_app - I_sch * (R_series + R_plug + R_disc) - V_sch

def dN_disc_dt(N_disc, T, E):
    """Eq.(9): Return time derivative of N_disc
    """
    I_ion = 2 * A * z_vo * e * (0.5*(N_disc+N_plug)) * a * nu_0 * \
        math.exp(-del_WA*eV_to_J / (kB * T)) * math.sinh((a * z_vo * e * E)/(2 * kB * T))
    return (1 / (z_vo * e * A * l_disc) * I_ion)

# Device Properties
l_plug = l_cell - l_disc # m - Length of the plug
A = const.pi * r_fil**2 # m^2 - Cross sectional area of flament
R_plug = l_plug / (e * z_vo * N_plug * mu_n * A) # Ohm - Plug resistivity

# Measurement Settings
V_max = 1 # V - Max sweep value
V_min = -1.4 # V - Min sweep value

t = np.linspace(0, V_max/sweeprate, 10000) # s - Sweep time for 0-1 V
del_t = t[1] - t[0] # sec
V_set = np.concatenate((t*sweeprate, np.flipud(t*sweeprate)))
t = np.linspace(0, V_min/sweeprate, -V_min/(sweeprate*del_t)) # s
V_reset = np.concatenate((t*sweeprate, np.flipud(t*sweeprate)))
V_sweep = np.concatenate((V_set, V_reset))   # V

# Initials
R_disc = np.zeros([len(V_sweep)])
N_disc = np.ones([len(V_sweep)]) * N_disc_min
T = np.ones([len(V_sweep)]) * T_0
V_sch = np.zeros([len(V_sweep)])
I_sch = np.zeros([len(V_sweep)])

# Simulation Parameters
start_step = 0
stop_step = 50000

# Do the voltage sweep
for i, V_app in enumerate(V_sweep):
    print("\n-----------------\nIteration: %d\n" %i)
    
    #1. Solve KCL for V_sch
    if i==0:
        R_disc[i] = l_disc / (e * z_vo * N_disc_min * mu_n * A) # Ohm - Disc resistivity
        V_sch[i] = optimize.newton(eq_V_sch, V_sch[i-1], args=(V_app, T[0], N_disc_min, R_disc[i], R_plug))
    else:
        try:
            # NOTE: I don't know why V_sch[i-2] converges but not V_sch[i-1].
            V_sch[i] = optimize.newton(eq_V_sch, V_sch[i-2], args=(V_app, T[i-1], N_disc[i-1], R_disc[i-1], R_plug), maxiter=100)
        except:
            V_sch[i] = V_sch[i-1]

    #2. Update I_sch, R_disc and E.
    R_disc[i] = l_disc / (e * z_vo * N_disc[i-1] * mu_n * A) # Ohm - Disc resistivity
    
    if V_app < 0: # RESET case
        e_phi_Bn = e_phi_Bn0 - e * J_to_eV * pow((pow(e, 3) * z_vo * N_disc[i-1] * ((e_phi_Bn0*eV_to_J/e) - (e_phi_n*eV_to_J/e) + V_sch[i]))/ \
            (8 * math.pi**2 * pow(eps_phi_B,3)) , 1/4) # eV -  Effective barrier height (V < 0)
        
        print("e_phi_Bn: %.2e" %e_phi_Bn)

        I_sch[i] = -A * A_star * T[i-1]**2 * math.exp(-(e_phi_Bn * eV_to_J) / (kB * T[i-1])) * \
            (math.exp(e * (-V_sch[i] / (kB * T[i-1]))) - 1) # A - Current passing device

        V_plug = R_plug * I_sch[i] # V
        V_disc = R_disc[i] * I_sch[i] # V
        E = (V_sch[i] + V_disc + V_plug) / l_cell # V/m - E_field
    
    else: # SET case
        W_00 = (e * h) * math.sqrt((z_vo * N_disc[i-1])/(mu_star * eps)) / (4 * math.pi) # J
        W_0 = W_00 * coth(W_00 / (kB * T[i-1])) # J
        xi = (W_00) / ((W_00/(kB * T[i-1])) - math.tanh(W_00 / (kB * T[i-1]))) # J
        print("W_00: %.2e\nW_0: %.2e\nxi: %.2e" %(W_00, W_0, xi))

        e_phi_Bn = e_phi_Bn0 - e * J_to_eV * pow((pow(e, 3) * z_vo * N_disc[i-1] * ((e_phi_Bn0*eV_to_J/e) - (e_phi_n*eV_to_J/e) + V_sch[i]))/ \
            (8 * math.pi**2 * pow(eps_phi_B,3)) , 1/4) # eV -  Effective barrier height (V < 0)
        
        I_sch[i] = A * A_star * (T[i-1]/kB) * \
            math.sqrt(math.pi * W_00 * e * (V_sch[i] + ((e_phi_Bn0 * eV_to_J / e) / (pow(math.cosh((W_00) / (kB * T[i-1])), 2) )))) * \
            math.exp((-e_phi_Bn*eV_to_J)/(W_0)) * (math.exp(e * V_sch[i] / xi) - 1)

        V_plug = R_plug * I_sch[i] # V 
        V_disc = R_disc[i] * I_sch[i] # V
        E = V_disc / l_disc # V/m

    # 3. Update T 
    T[i] = (V_sch[i] + V_disc + V_plug) * I_sch[i] * R_th_eff + T_0 # K
    
    if T[i] > 500: # To get rid of single outlier T due to unstable Newton method.
        T[i] = T[i-1]

    # 4. Solve ODE for N_disc
    N_disc[i] = N_disc[i-1] + del_t * dN_disc_dt(N_disc[i-1], T[i], E)
 
    # Limit N_disc_min < N_disc < N_disc_max 
    if N_disc[i] < N_disc_min: N_disc[i] = N_disc_min 
    if N_disc[i] > N_disc_max: N_disc[i] = N_disc_max 
 
    # Display 
    print("V_app: %.2e\nV_sch: %.2e\nV_disc: %.2e\nV_plug: %.2e\nR_disc: %.2e\nI_sch: %.2e\nT: %.2f\nN_disc: %.2e\nE: %.2e" \
            %(V_app, V_sch[i], V_disc, V_plug, R_disc[i], I_sch[i], T[i], N_disc[i], E)) 
    
# Plotting
I_sch = np.abs(I_sch)
plt.semilogy(V_sweep,I_sch)
plt.grid()
plt.xlabel("Applied Voltage (V)")
plt.ylabel("Current (A)")
plt.ylim([1e-8, 1e-3])
plt.show()
