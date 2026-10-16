import os
import numpy as np
import matplotlib.pyplot as plt
import nanowire
import matplotlib.colors as colors

TWO_PI = 2 * np.pi

trapping_freq = 0.6e12


def trap_frequency_factor(factor=1.0):
    trap_freqs = np.array([trapping_freq, 
                            15e12, 
                            (1/10)*(trapping_freq*factor)])
    return trap_freqs


def dot_centres_factor(factor=1.0):
    centre_distance = factor * 1000e-9
    dot_centres = np.array([-centre_distance, 0, centre_distance])
    return dot_centres


def grid_search(n_electrons=8, 
                n_steps=10, 
                trap_min_factor=0.8, 
                trap_max_factor=1.2,
                dot_min_factor=0.8, 
                dot_max_factor=1.2):
    trap_factors = np.linspace(trap_min_factor, trap_max_factor, n_steps)
    dot_factors = np.linspace(dot_min_factor, dot_max_factor, n_steps)
    results = [[] for _ in range(n_steps)]

    for i, tf in enumerate(trap_factors):
        for j, dc in enumerate(dot_factors):
            trap_freqs = trap_frequency_factor(tf)
            dot_centres = dot_centres_factor(dc)

            wire = nanowire.Nanowire(n_electrons=n_electrons, 
                                     trap_freqs=trap_freqs, 
                                     dot_freqs=np.array([trapping_freq, 15e12, trapping_freq]), 
                                     dot_centres=dot_centres,
                                     verbose=False)
            
            g_detuning_ratio = wire.max_g_over_detuning
            results[i].append(g_detuning_ratio)
            print(f"trap freq factor: {tf:.3f}, dot centre factor: {dc:.3f} => max g/detuning: {g_detuning_ratio:.3e}  ", end="\r")
    
    return results


def plot_g_over_detuning_grid(n_electrons, 
                              results, 
                              trap_min_factor=0.8, 
                              trap_max_factor=1.2, 
                              dot_min_factor=0.8, 
                              dot_max_factor=1.2,
                              mark_points=None, 
                              marker_size=100,
                              savefig=False):
    data = np.array(results)

    vmin = data.min()
    vmax = data.max()
    vcentre = 0.1   # the value that should have a fixed colour

    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcentre, vmax=vmax)

    # cmap = colors.LinearSegmentedColormap.from_list(
    #     "green_red_custom",
    #     [
    #         (0.0,  "#ccffcc"),   # light green (lowest values)
    #         (0.4999999,  "green"),     # pure green at 0.1
    #         (0.5000001, "#ff9999"),   # light red above 0.1
    #         (1.0,  "darkred")    # dark red at max
    #     ]
    # )

    cmap = colors.LinearSegmentedColormap.from_list(
        "green_red_custom",
        [
            (0.0,  "#ccffcc"),   # light green (lowest values)
            (0.4999999,  "green"),     # pure green at 0.1
            (0.5000001, "gray"),   # light red above 0.1
            (1.0,  "black")    # dark red at max
        ]
    )

    plt.imshow(data,
        # extent=(0, 
        #         data.shape[0]-1, 
        #         data.shape[1]-1, 
        #         0),
        # extent=(min_factor, 
        #         max_factor, 
        #         min_factor, 
        #         max_factor),
        # extent=(trap_frequency_factor(trap_min_factor)[-1] * 10**-9, 
        #         trap_frequency_factor(trap_max_factor)[-1] * 10**-9, 
        #         2 * dot_centres_factor(dot_min_factor)[-1] * 10**6, 
        #         2 * dot_centres_factor(dot_max_factor)[-1] * 10**6),
        extent=(2 * dot_centres_factor(dot_min_factor)[-1] * 10**6, 
                2 * dot_centres_factor(dot_max_factor)[-1] * 10**6,
                trap_frequency_factor(trap_min_factor)[-1] * 10**-9 / TWO_PI, 
                trap_frequency_factor(trap_max_factor)[-1] * 10**-9 / TWO_PI),
        origin='lower',
        aspect='auto',
        norm=norm,
        cmap=cmap) 

    # Create colorbar
    cbar = plt.colorbar(#label='max coupling/detuning', 
                        ticks=[vmin, vcentre, vmax],
                        cmap=cmap)
    cbar.ax.set_yticklabels([f"{vmin:.2f}", f"{vcentre:.3f}", f"{vmax:.2f}"])

    if mark_points is not None:
        for mark_point in mark_points:
            x_mark, y_mark = mark_point
            plt.scatter(
                # trap_frequency_factor(x_mark)[-1] * 10**-9,
                2 * dot_centres_factor(y_mark)[-1] * 10**6,
                trap_frequency_factor(x_mark)[-1] * 10**-9 / TWO_PI,
                # y_mark,
                # x_mark,
                s=marker_size,
                facecolors='none',
                edgecolors='blue',
                linewidths=2,
                zorder=5
                )
            
    plt.ylabel('Trapping frequency $\omega_y/2\pi$ (GHz)')
    plt.xlabel(f'$d = | \mathbf{{r}}_b - \mathbf{{r}}_a |$ ($\\mathrm{{\\mu}}$m)')
    plt.title('$g_{y,12}/\Delta_2$ (phonon mode 2)')
    if savefig:
        plt.savefig(f"plots/parameter_sweep_g_over_detuning_n_{n_electrons}.pdf", bbox_inches='tight', dpi=300)
    plt.show()


def parameter_search(n_electrons=8, 
                     n_steps=30, 
                     trap_min_factor=0.8, 
                     trap_max_factor=1.2, 
                     dot_min_factor=0.8, 
                     dot_max_factor=1.2, 
                     show_plot=True):
    pwd = os.getcwd() 
    filepath = os.path.join(pwd, f"data/n_electrons_{n_electrons}_n_steps_{n_steps}_trap_min_{trap_min_factor}_max_{trap_max_factor}_dot_min_{dot_min_factor}_max_{dot_max_factor}.csv")

    if os.path.exists(filepath):
        results = np.loadtxt(filepath, delimiter=",")
    else:
        results = grid_search(n_electrons=n_electrons, 
                            n_steps=n_steps, 
                            trap_min_factor=trap_min_factor, 
                            trap_max_factor=trap_max_factor,
                            dot_min_factor=dot_min_factor,
                            dot_max_factor=dot_max_factor)
        np.savetxt(filepath, results, delimiter=",")
    if show_plot:
        plot_g_over_detuning_grid(n_electrons, 
                                  results, 
                                  trap_min_factor=trap_min_factor, 
                                  trap_max_factor=trap_max_factor, 
                                  dot_min_factor=dot_min_factor, 
                                  dot_max_factor=dot_max_factor)
    else:
        print(f"Completed parameter search without plotting. -- ({filepath})")
    return results