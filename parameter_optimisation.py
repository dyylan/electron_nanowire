import os
import numpy as np
import nanowire
import parameter_search as ps


def parameter_optimisation(n_electrons=8, 
                           n_steps=30, 
                           trap_min_factor=0.8, 
                           trap_max_factor=1.2, 
                           dot_min_factor=0.8,
                           dot_max_factor=1.2,
                           g_over_detuning_filter=(0.08, 0.1)):
    pwd = os.getcwd() 
    filepath = os.path.join(pwd, f"data/n_electrons_{n_electrons}_n_steps_{n_steps}_trap_min_{trap_min_factor}_max_{trap_max_factor}_dot_min_{dot_min_factor}_max_{dot_max_factor}.csv")
    if os.path.exists(filepath):
        results = np.loadtxt(filepath, delimiter=",")
    else:
        raise ValueError("No existing data file found. Please run grid_search first to generate data.")
    trap_factors = np.linspace(trap_min_factor, trap_max_factor, n_steps)
    dot_factors = np.linspace(dot_min_factor, dot_max_factor, n_steps)
    Js = []
    g_over_detunings = []
    indices = []
    boundary_g_over_indices = []
    for i in range(n_steps):
        for j in range(n_steps):
            g_over_detuning = results[i,j]
            if (g_over_detuning_filter[0] <= g_over_detuning) and (g_over_detuning <= g_over_detuning_filter[1]):
                trap_freq = ps.trap_frequency_factor(factor=trap_factors[i])
                dot_centres = ps.dot_centres_factor(factor=dot_factors[j])
                wire = nanowire.Nanowire(n_electrons=n_electrons, 
                                         trap_freqs=trap_freq,
                                         dot_centres=dot_centres,
                                         verbose=False)
                Js.append(max(abs(wire.interactions[0])))
                g_over_detunings.append(g_over_detuning)
                indices.append((i, j))
            if 0.093 <= g_over_detuning <= 0.1:
                boundary_g_over_indices.append((i, j))
    Js = np.array(Js)
    J = np.max(Js)
    i_max, j_max = indices[np.argmax(Js)]
    g_over_detuning = g_over_detunings[np.argmax(Js)]
    print(f"For n={n_electrons} -- g/detuning: {g_over_detuning:.3f}, J: {J:.3e}, Trap Freq Factor: {trap_factors[i_max]:.3f}, Dot Centre Factor: {dot_factors[j_max]:.3f} [point ({i_max}, {j_max})]")            
    
    trap_freq = ps.trap_frequency_factor(factor=trap_factors[i_max])
    dot_centres = ps.dot_centres_factor(factor=dot_factors[j_max])  
    print(f"\t --- Trap Freq: {trap_freq[-1]:.3e}, Dot Centre Factor: {2*dot_centres[-1]:.3e}")            

    # ps.plot_g_over_detuning_grid(n_electrons, results, min_factor, max_factor, mark_points=[(factors[i_max], factors[j_max])], marker_size=100)
    ps.plot_g_over_detuning_grid(n_electrons, 
                                 results, 
                                 trap_min_factor, 
                                 trap_max_factor, 
                                 dot_min_factor,
                                 dot_max_factor,
                                 mark_points=[(trap_factors[i_max], dot_factors[j_max])],
                                 marker_size=100,
                                 savefig=True)

    mark_points = []
    for boundary_i, boundary_j in boundary_g_over_indices:
        mark_points.append((trap_factors[boundary_i], dot_factors[boundary_j]))
    if len(mark_points) > 0:
        ps.plot_g_over_detuning_grid(n_electrons, 
                                     results, 
                                     trap_min_factor, 
                                     trap_max_factor, 
                                     dot_min_factor, 
                                     dot_max_factor, 
                                     mark_points=mark_points, 
                                     marker_size=100)

    wire = nanowire.Nanowire(n_electrons=n_electrons, 
                                trap_freqs=trap_freq,
                                dot_centres=dot_centres, 
                                verbose=False) 
    wire.plot_nanowire_positions(plot_potentials=True, savefig=False)
    wire.plot_phonon_modes(savefig=False, xrange=[0e12, 0.65e12])
    wire.plot_phonon_eigenmodes(savefig=False)
    wire.plot_interactions()
