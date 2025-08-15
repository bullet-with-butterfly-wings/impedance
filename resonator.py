import numpy as np

w = 2*np.pi * 50e6
d0 = 2e-3  # diameter of the wire
#dimensions
d = 1 #diameter of coil
b = 1 #height
D = 1 #diameter of shield
t = 1 #pitch
#inherent impedances
Z_load = 50
C_trap = 0.1e-12
C_wires = 0.1e-12
R_wires = 0.1 
R_trap = 0.1
#factors
K_cs = 39.37*0.75*1e-12*1/(np.log(D/d))
K_lc = (b*39.37*0.025*(d**2)*(1-(d/D)**2)*1e-6 )/(t**2)
H = 11.26*b/d+8+27/np.sqrt(b/d)
#electric
C_coil = (H*d)*1e-12 
L_coil = b*K_lc
C_shield = b*K_cs
#
L_antenna = 0.1 #can be adjusted, impedance matching

Z_input = L_antenna * (1j*w + L_coil*w**2/(1j*2*L_coil + Z_load))
#Calculate b if you have d, D, t
b = ((C_trap + C_wires + d*35e-12)/(K_cs + 11.26e-12))*(np.sqrt((K_cs+11.26e-12)/(K_lc*(C_trap + C_wires + d*35e-12)**2*w**2) +1/4)-1/2)
#calculate N if you have b, d, D
N = b /np.sqrt(16*np.pi**2 * (D-d)**4 / b**2 - np.pi**2 * d **2 )
t = N/b
#resistances
R_coil = 1.68e-8*np.sqrt(N * np.sqrt(np.pi**2 * D**2 + t**2))/(np.pi * d0 * 10e-6)
R_shield = (N* 1.68e-8 * np.sqrt(np.pi**2 * D**2 + t**2))/(b * 10e-6)
#at (16) i am not sure what they meant...
C_coil = 
a = C_trap / (C_shield + C_wires)
w_res = 1/np.sqrt(L_coil * (C_shield + C_trap + C_wires + C_coil))
Q = w_res * L_coil / (R_coil + R_shield + R_wires + (a/a+1) * R_trap)
