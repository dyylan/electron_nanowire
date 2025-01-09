import nanowire

# config_InAs = {
#     trap_freqs = np.array([1e12, 15e12, (1/10)*1e12]),
#     dot_freqs = np.array([1e12, 15e12, 1e12]),
#     dot_centres = np.array([-750e-9, 0, 750e-9]),
#     effective_mass=0.067,
#     omega_0 = np.ones(1) * (4e+11),
#     alpha_R = 0.3e-10  
# }

if __name__=="__main__":
    wire = nanowire.Nanowire(8)
    # print(f"wire frequencies = {wire.eigenfrequencies}")
    # print(f"wire modes = {wire.eigenmodes}")
    # print(f"wire z positions = {wire.positions}")
    wire.plot_nanowire_positions(plot_potentials=True)
    wire.plot_nanowire_positions_2d(plot_potentials=True)
    # wire.plot_spin_interactions()
    # print(wire.interactions)
    # print(wire.eigenfrequencies)
    # wire.plot_phonon_modes(xrange=[0.36e13, 0.51e13], savefig=True)
    wire.plot_phonon_modes()
    wire.plot_interactions()