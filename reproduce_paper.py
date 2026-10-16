"""Reproduce the headline numbers of arXiv:2603.13519 (N = 8, dots at +/-1000 nm) and print
them with explicit units, together with the mode-resolved table requested by the referee.

All quantities inside `Nanowire` are angular frequencies (rad/s); here they are also
converted to cyclic frequencies (/2pi) for quoting in MHz/GHz.

Run:  python3 reproduce_paper.py [alpha_eff_in_eV_m]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import nanowire

TWO_PI = 2 * np.pi
alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 2e-12   # eV m (value used for the paper)

w = nanowire.Nanowire(8, alpha_R=alpha, verbose=False)
f = w.eigenfrequencies
m2 = np.argsort(f)[1]                     # axial mode 2 (the one used in the paper)
w0 = w.omega_0[0]
B = w0 * nanowire.Nanowire.hbar / (nanowire.Nanowire.ge * nanowire.Nanowire.mub)

print(f"alpha_eff                 = {alpha:.2e} eV m")
print(f"omega_0                   = {w0:.4e} rad/s  =  {w0/TWO_PI/1e9:.3f} GHz  (B = {B:.3f} T, g = {nanowire.Nanowire.ge})")
print(f"omega_y,2                 = {f[m2]:.5e} rad/s  =  {f[m2]/TWO_PI/1e9:.4f} GHz")
print(f"Delta_2 = omega_0-omega_2 = {w.detunings[m2]:.4e} rad/s  =  {w.detunings[m2]/TWO_PI/1e9:.4f} GHz")
print(f"b_y,12                    = {w.eigenmodes[0,2,m2]:+.5f}   b_y,N2 = {w.eigenmodes[-1,2,m2]:+.5f}")
print(f"g_y,12                    = {w.g[0,m2]:.4e} rad/s  =  {w.g[0,m2]/TWO_PI/1e6:.2f} MHz")
print(f"g_y,12/Delta_2            = {w.g[0,m2]/w.detunings[m2]:.4f}")
print(f"J_1N (mode 2 only)        = {w.J_per_mode[m2]:.4e} rad/s  =  {w.J_per_mode[m2]/TWO_PI/1e6:.3f} MHz")
print(f"J_1N (all modes)          = {w.J_1N:.4e} rad/s  =  {w.J_1N_cyclic/1e6:.3f} MHz")
print(f"iSWAP time pi/(4|J|)      = {w.iswap_time()*1e9:.1f} ns")
print()
w.mode_table(n_modes=8)
