import nanowire


def plots(n_electrons=8):
    wire = nanowire.Nanowire(n_electrons=n_electrons)

    wire.plot_nanowire_positions(plot_potentials=True, savefig=True)
    wire.plot_nanowire_positions_2d(plot_potentials=True, savefig=True)

    wire.plot_phonon_modes(savefig=True, xrange=[0e12, 0.65e12])
    wire.plot_phonon_eigenmodes(savefig=True)
    wire.plot_interactions()