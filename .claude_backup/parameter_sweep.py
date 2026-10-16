import nanowire
import parameter_search as ps
import parameter_optimisation as po
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# n_electrons_list = [6,7,8,9,10]
# n_electrons_list = [10]
n_electrons_list = []

if __name__=="__main__":
    for n_electrons in n_electrons_list:
        n_steps = 50
        if n_electrons < 8:
            trap_min_factor = 0.6 * (2 * 8.0 / (n_electrons+8.0))
            trap_max_factor = 1.0 * (2 * 8.0 / (n_electrons+8.0))
            print(f"--- n={n_electrons}: trap_min_factor={trap_min_factor:.3f}, trap_max_factor={trap_max_factor:.3f} --- ")
            dot_max_factor = 1/(0.9 * (2 * 8.0 / (n_electrons+8.0)))
            dot_min_factor = 1/(1.3 * (2 * 8.0 / (n_electrons+8.0)))
            print(f"--- n={n_electrons}: dot_min_factor={dot_min_factor:.3f}, dot_max_factor={dot_max_factor:.3f} --- ")
        elif n_electrons == 8:
            trap_min_factor = 0.7 * (2 * 8.0 / (n_electrons+8.0))
            trap_max_factor = 1.1 * (2 * 8.0 / (n_electrons+8.0))
            print(f"--- n={n_electrons}: trap_min_factor={trap_min_factor:.3f}, trap_max_factor={trap_max_factor:.3f} --- ")
            dot_max_factor = 1/(0.85 * (2 * 8.0 / (n_electrons+8.0)))
            dot_min_factor = 1/(1.25 * (2 * 8.0 / (n_electrons+8.0)))
            print(f"--- n={n_electrons}: dot_min_factor={dot_min_factor:.3f}, dot_max_factor={dot_max_factor:.3f} --- ")
        else:
            trap_min_factor = 0.8 * (2 * 8.0 / (n_electrons+8.0))
            trap_max_factor = 1.2 * (2 * 8.0 / (n_electrons+8.0))
            print(f"--- n={n_electrons}: trap_min_factor={trap_min_factor:.3f}, trap_max_factor={trap_max_factor:.3f} --- ")
            dot_max_factor = 1/(0.8 * (2 * 8.0 / (n_electrons+8.0)))
            dot_min_factor = 1/(1.2 * (2 * 8.0 / (n_electrons+8.0)))
            print(f"--- n={n_electrons}: dot_min_factor={dot_min_factor:.3f}, dot_max_factor={dot_max_factor:.3f} --- ")

        ps.parameter_search(n_electrons=n_electrons, 
                            n_steps=n_steps, 
                            trap_min_factor=trap_min_factor, 
                            trap_max_factor=trap_max_factor,
                            dot_min_factor=dot_min_factor,
                            dot_max_factor=dot_max_factor,
                            show_plot=True)
        po.parameter_optimisation(n_electrons=n_electrons, 
                                  n_steps=n_steps, 
                                  trap_min_factor=trap_min_factor, 
                                  trap_max_factor=trap_max_factor,
                                  dot_min_factor=dot_min_factor,
                                  dot_max_factor=dot_max_factor)
        
    n = [6, 7, 8, 9, 10]
    optimal_Js = np.array([35.80e6, 40.43e6, 43.45e6, 45.92e6, 50.80e6])/1e6

    f = [49.54, 50.42, 54.24, 56.70, 54.86]
    d = [1.554, 1.704, 1.846, 1.970, 2.066]


    plt.figure()
    plt.plot(n, optimal_Js, marker='o', color='blue') 

    offsets = [(17, -8), (11, -16), (11, -16), (10, -15), (-90, -15)]  # above, below

    for i, (ni, Ji, fi, di) in enumerate(zip(n, optimal_Js, f, d)):
        label = f"$\omega_y={fi:.2f}$ GHz\n$d={di:.3f}~\mathrm{{\mu m}}$"
        dx, dy = offsets[i]

        plt.annotate(
            label,
            (ni, Ji),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="left",
            fontsize=12,
            # bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8)
        )
    
    plt.xlabel("$N$")
    plt.ylabel("Maximum $J_{1 N}$ (MHz)")
    # plt.title("Optimal J vs n")

    # force integer ticks on x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.grid(False)
    plt.tight_layout()
    plt.savefig("plots/optimal_J_vs_n_electrons.pdf", dpi=300)
    plt.show()