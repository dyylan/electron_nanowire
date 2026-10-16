import numpy as np
import nanowire


config = {
    7 : {
        'trap_freqs' : np.array([0.6e12, 15e12, (1/10)*0.6*(8/7.5)*1e12]),
        'dot_freqs' : np.array([0.6e12, 15e12, 0.6e12]),
        # 'dot_centres' : np.array([-857e-9, 0, 857e-9]), 
        'dot_centres' : np.array([-900e-9, 0, 900e-9]),       
    },
    8 : {
        'trap_freqs' : np.array([0.6e12, 15e12, (1/10)*0.6e12]),
        'dot_freqs' : np.array([0.6e12, 15e12, 0.6e12]),
        'dot_centres' : np.array([-1000e-9, 0, 1000e-9]),
    },
    9 : {
        'trap_freqs' : np.array([0.6e12, 15e12, (1/10)*0.6*(8/8.5)*1e12]),
        'dot_freqs' : np.array([0.6e12, 15e12, 0.6e12]),
        'dot_centres' : np.array([-1080e-9, 0, 1080e-9]),
    },
    10 : {
        'trap_freqs' : np.array([0.6e12, 15e12, (1/10)*0.6*(8/9)*1e12]),
        'dot_freqs' : np.array([0.6e12, 15e12, 0.6e12]),
        'dot_centres' : np.array([-1170e-9, 0, 1170e-9]),
    },
}

if __name__=="__main__":

    ns = [9, 10]

    for n in ns:
        trap_freqs = config[n]['trap_freqs']
        dot_freqs = config[n]['dot_freqs']
        dot_centres = config[n]['dot_centres']

        wire = nanowire.Nanowire(n_electrons=n, 
                                 trap_freqs=trap_freqs, 
                                 dot_freqs=dot_freqs, 
                                 dot_centres=dot_centres)

        wire.plot_nanowire_positions(plot_potentials=True, savefig=True)
        wire.plot_nanowire_positions_2d(plot_potentials=True, savefig=True)

        wire.plot_phonon_modes(savefig=True, xrange=[0e12, 0.65e12])
        wire.plot_phonon_eigenmodes(savefig=True)
        wire.plot_interactions()