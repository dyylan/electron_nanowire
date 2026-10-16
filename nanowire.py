import autograd as ag
from autograd import numpy as agnp
import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt


plt.rc("text", usetex=True)
font = {"family": "serif", "size": 16, "serif": ["computer modern roman"]}
plt.rc("font", **font)

fontsize_title = 24
fontsize_axis = 22
fontsize_legend = 18
fontsize_ticks = 18
figure_size = [10, 8]


class Nanowire:
    """Phonon bus of a gate-defined electron chain between two quantum dots.

    UNITS: every frequency-like quantity inside this class is an ANGULAR
    frequency in rad/s (trap frequencies, eigenfrequencies, omega_0, g, J,
    detunings). Energies are expressed as angular frequencies via E/hbar.
    Use the *_cyclic properties (or divide by 2*pi) to obtain ordinary
    frequencies in Hz for quoting in MHz/GHz.
    """

    TWO_PI = 2 * np.pi
    k = 8.98755179e9 / 12.9  # coulomb constant / relative permittivity in GaAs 
    hbar = 1.054571817e-34  # reduced planck constant
    e = 1.602176634e-19  # charge of electron
    m_e = 9.1093837e-31 # electron mass
    l = 1 # length scale (was 1e-6 before)
    ge = 0.39 # electron g-factor in GaAs quantum dot
    mub = 9.2740100783e-24 # Bohr Magneton
    c = 299792458 # speed of light
    trapping_freq = 0.6e12  # trapping frequency in rad/s


    def __init__(self, 
        n_electrons, 
        trap_freqs = np.array([trapping_freq, 15e12, (1/10)*trapping_freq]),
        dot_freqs = np.array([trapping_freq, 15e12, trapping_freq]),
        dot_centres = np.array([-1000e-9, 0, 1000e-9]),
        effective_mass=0.067,
        # omega_0 = np.ones(1) * (1.765e+11),  # Zeeman splitting, rad/s (B = 5.151 T, g = 0.39)
        omega_0 = np.ones(1) * (1.78687e+11),  # Zeeman splitting, rad/s (B = 5.21 T, g = 0.39)
        alpha_R = 2e-12,  # effective linear spin-orbit coefficient alpha_eff = alpha + beta, eV m.
                          # 2e-12 eV m is at the upper end of gated GaAs/AlGaAs wells with the
                          # chain along [110] (alpha ~ beta ~ 4-5e-13 eV m each, see e.g.
                          # Dettwiler et al., PRX 7, 031010 (2017)). NOTE: this is the value
                          # used for all paper results; the manuscript text must quote it.
        coupling_axes = np.array([1, 0, 1]),  # which displacement components couple to the spin:
                                              # index 0 = transverse x, 1 = z (2DEG), 2 = axial y
        verbose=True,
        ):
        self.coupling_axes = np.asarray(coupling_axes, dtype=float)
        self.verbose = verbose
        self.n_electrons = n_electrons
        self.trap_freqs = trap_freqs
        self.dot_freqs = dot_freqs
        self.dot_centres = dot_centres
        self.effective_mass = effective_mass
        self.mass = effective_mass * Nanowire.m_e
        self.omega_0 = omega_0
        self.alpha_R = (alpha_R * Nanowire.e / Nanowire.hbar) * np.reshape(np.array([1] + [0 for _ in range(n_electrons - 2)] + [1]), (self.n_electrons, 1))
        self.positions = self._calculate_equilibrium_positions()
        self.hessian = self._calculate_equilibrium_hessian()
        self.eigenfrequencies, self.eigenmodes = self._calculate_normal_mode_eigenpairs()
        self.interactions = self._spin_interaction_graph()

    def nondimensionalize_decorator(expr):
        def _expr(x, n, m, omega, dot_omega, pos, l):
            return expr(x * l, n, m, omega, dot_omega, pos * l, l) * l / (Nanowire.k * Nanowire.e ** 2) 

        return _expr

    def flatten_decorator(expr):
        def _expr(x, n, m, omega, dot_omega, pos, l):
            return expr(x.reshape(n, 3), n, m, omega, dot_omega, pos, l)

        return _expr

    # nondimensionalized coulomb potential
    @staticmethod
    @nondimensionalize_decorator
    def expr_coul(x, n, m, omega, dot_omega, pos, l):
        combs = np.triu_indices(n, k=1)
        return (
            Nanowire.k
            * Nanowire.e ** 2
            / agnp.linalg.norm(x[combs[0]] - x[combs[1]], axis=-1)
        ).sum()

    # nondimensionalized harmonic potential
    @staticmethod
    @nondimensionalize_decorator
    def expr_har(x, n, m, omega, dot_omega, pos, l):
        x_1 = pos[0] * np.pad(np.ones(n).reshape(n, 1),pad_width=[[0, 0], [2, 0]])
        x_nanowire = pos[1] * np.pad(np.ones(n).reshape(n, 1),pad_width=[[0, 0], [2, 0]])
        x_n = pos[2] * np.pad(np.ones(n).reshape(n, 1),pad_width=[[0, 0], [2, 0]])
        dot1_potential = m * dot_omega ** 2 * (x - x_1) ** 2 / 2
        nanowire_potential = m * omega ** 2 * (x - x_nanowire) ** 2 / 2
        dotn_potential = m * dot_omega ** 2 * (x - x_n) ** 2 / 2
        potentials = np.array([dot1_potential, nanowire_potential, dotn_potential]).min(axis=0)
        # return (m * omega ** 2 * x ** 2 / 2).sum()
        return potentials.sum()

    # flattened nondimensionalized total potential
    @staticmethod
    @flatten_decorator
    def expr_tot(x, n, m, omega, dot_omega, pos, l):
        return (Nanowire.expr_coul(x, n, m, omega, dot_omega, pos, l) + Nanowire.expr_har(x, n, m, omega, dot_omega, pos, l)) 

    # hessian of flattened nondimensionalized total potential
    @staticmethod
    def hess_expr_tot(x, n, m, omega, dot_omega, pos, l):
        return ag.hessian(Nanowire.expr_tot, 0)(x, n, m, omega, dot_omega, pos, l)

    # equilibrium position
    def _calculate_equilibrium_positions(self):
        op_powell = opt.minimize(
                Nanowire.expr_tot,
                np.pad(
                    # np.linspace(
                    #     -(self.n_electrons - 1) / 2, (self.n_electrons - 1) / 2, self.n_electrons
                    # ).reshape(self.n_electrons, 1),
                    np.linspace(
                        self.dot_centres[0], self.dot_centres[2], self.n_electrons
                    ).reshape(self.n_electrons, 1),
                    pad_width=[[0, 0], [2, 0]],
                ).flatten(),
                args=(self.n_electrons, self.mass, self.trap_freqs, self.dot_freqs, self.dot_centres, Nanowire.l),
                method="Powell",
                options={"maxiter": 1000, "disp": True if self.verbose else False},
                tol=1e-15,
            )
        op_cg = opt.minimize(
                Nanowire.expr_tot,
                np.pad(
                    # np.linspace(
                    #     -(self.n_electrons - 1) / 2, (self.n_electrons - 1) / 2, self.n_electrons
                    # ).reshape(self.n_electrons, 1),
                    np.linspace(
                        self.dot_centres[0], self.dot_centres[2], self.n_electrons
                    ).reshape(self.n_electrons, 1),
                    pad_width=[[0, 0], [2, 0]],
                ).flatten(),
                args=(self.n_electrons, self.mass, self.trap_freqs, self.dot_freqs, self.dot_centres, Nanowire.l),
                method="CG",
                options={"maxiter": 1000, "disp": True if self.verbose else False},
                tol=1e-15,
            )

        def op_val(op):
            return Nanowire.expr_tot(op.x, self.n_electrons, self.mass, self.trap_freqs, self.dot_freqs, self.dot_centres, Nanowire.l)

        if op_val(op_powell) < op_val(op_cg):
            op = op_powell
            if self.verbose:
                print("Using Powell optimizer result.")
        else:
            op = op_cg
            if self.verbose:
                print("Using CG optimizer result.")
        x_ep = op.x.reshape(self.n_electrons, 3)
        return x_ep

    # hessian evaluated at equilibrium position
    def _calculate_equilibrium_hessian(self):
        return Nanowire.hess_expr_tot(
            self.positions.flatten() / Nanowire.l, self.n_electrons, self.mass, 
            self.trap_freqs, self.dot_freqs, self.dot_centres, Nanowire.l
        )

    # normal mode eigenpairs
    def _calculate_normal_mode_eigenpairs(self):
        lamb, b = np.linalg.eigh(self.hessian)
        eigenfrequencies = np.sqrt(
            lamb * Nanowire.k * Nanowire.e ** 2 / (Nanowire.l ** 3 * self.mass)
        )  # eigenfrequencies
        eigenmodes = b.reshape(self.n_electrons, 3, 3 * self.n_electrons)  # eigenvectors
        # print(eigenmodes)
        return eigenfrequencies, eigenmodes

    # Calculates spin interaction graph
    def _spin_interaction_graph(self):
        eta = np.einsum(
            "ikm,k,m->im",
            self.eigenmodes,
            self.coupling_axes,
            np.sqrt(Nanowire.hbar * self.mass * self.eigenfrequencies / 2 ),
            # np.sqrt(self.mass * self.eigenfrequencies / 2 ),
        )
        g = np.einsum("im,in->imn", self.alpha_R, eta)
        Js = np.einsum(
            "ij,imn,jmn,n,mn->ij",
            1 - np.identity(self.n_electrons),
            g,
            g,
            self.eigenfrequencies,
            1 / np.subtract.outer(self.omega_0 ** 2, self.eigenfrequencies ** 2),
        ) / Nanowire.hbar # hbar comes with the time t 

        ratios = np.einsum(
            "imn,mn->in",
            g,
            1/(Nanowire.hbar*(np.subtract.outer(self.omega_0, self.eigenfrequencies)))
        )
        # detunings = [abs(self.omega_0[0] - freq) for freq in self.eigenfrequencies]
        if self.verbose:
            # print(f"eigenmodes = {self.eigenmodes}")
            print(f"max g = {g[0,0,1]/Nanowire.hbar}")
            print(f"max g e11 = {g[0,0,1]/Nanowire.hbar / 1e11}")
            print(f"g/detunings = {[abs(ratios[0,i]) for i in range(len(ratios[0]))]}")
            print(f"maximum g/detuning = {max(abs(ratios[0]))}")
            print(f"max g/detunings = {g[0,0,1]/Nanowire.hbar/(1.765e+11 - self.eigenfrequencies[1])}")
            print(f"approx J = {g[0,0,1]/Nanowire.hbar*g[self.n_electrons-1,0,1]/Nanowire.hbar* self.eigenfrequencies[1] / (self.omega_0[0]**2 - self.eigenfrequencies[1]**2)}")
            print(f"J = {max(abs(Js[0])) / Nanowire.hbar}")
        self.max_g_over_detuning = max(abs(ratios[0]))
        self.g = g[:, 0, :] / Nanowire.hbar          # g_{jm} in rad/s, shape (n_electrons, 3n modes)
        self.detunings = self.omega_0[0] - self.eigenfrequencies   # Delta_m = omega_0 - omega_m, rad/s
        self.J_per_mode = (self.g[0] * self.g[-1] * self.eigenfrequencies
                           / (self.omega_0[0] ** 2 - self.eigenfrequencies ** 2))  # J_1N^(m), rad/s
        return Js / Nanowire.hbar

    # ---- unit helpers (cyclic frequencies in Hz) -------------------------------
    @property
    def eigenfrequencies_cyclic(self):
        return self.eigenfrequencies / Nanowire.TWO_PI

    @property
    def omega_0_cyclic(self):
        return self.omega_0 / Nanowire.TWO_PI

    @property
    def interactions_cyclic(self):
        """J_{ij}/2pi in Hz."""
        return self.interactions / Nanowire.TWO_PI

    @property
    def J_1N(self):
        """Effective end-to-end XY coupling J_1N in rad/s (sum over all modes)."""
        return self.interactions[0, -1]

    @property
    def J_1N_cyclic(self):
        """J_1N/2pi in Hz."""
        return self.J_1N / Nanowire.TWO_PI

    def iswap_time(self):
        """Duration of an iSWAP for H = J (XX + YY): t = pi / (4 |J|), in seconds."""
        return np.pi / (4 * abs(self.J_1N))

    def mode_table(self, n_modes=None, axis=2, print_table=True):
        """Mode-resolved table requested by the referee: omega_m, b_1m, b_Nm, g_1m, g_Nm,
        Delta_m and J_1N^(m) for every mode (sorted by frequency). `axis` selects which
        displacement component of the eigenvector is tabulated (2 = axial y by default).
        Returns a list of dicts; frequencies are given both in rad/s and in Hz (/2pi)."""
        order = np.argsort(self.eigenfrequencies)
        if n_modes is not None:
            order = order[:n_modes]
        rows = []
        for m in order:
            rows.append(dict(
                mode=int(m),
                omega_rad_s=float(self.eigenfrequencies[m]),
                f_Hz=float(self.eigenfrequencies[m] / Nanowire.TWO_PI),
                b_1=float(self.eigenmodes[0, axis, m]),
                b_N=float(self.eigenmodes[-1, axis, m]),
                g_1_rad_s=float(self.g[0, m]),
                g_N_rad_s=float(self.g[-1, m]),
                Delta_rad_s=float(self.detunings[m]),
                g_over_Delta=float(self.g[0, m] / self.detunings[m]),
                J_rad_s=float(self.J_per_mode[m]),
            ))
        if print_table:
            hdr = (f"{'m':>3} {'w_m/2pi (GHz)':>14} {'b_1':>9} {'b_N':>9} "
                   f"{'g_1/2pi (MHz)':>14} {'g_N/2pi (MHz)':>14} {'D/2pi (GHz)':>12} "
                   f"{'g/D':>8} {'J^(m)/2pi (kHz)':>16}")
            print(hdr)
            for r in rows:
                print(f"{r['mode']:>3} {r['f_Hz']/1e9:>14.4f} {r['b_1']:>+9.4f} {r['b_N']:>+9.4f} "
                      f"{r['g_1_rad_s']/Nanowire.TWO_PI/1e6:>+14.3f} {r['g_N_rad_s']/Nanowire.TWO_PI/1e6:>+14.3f} "
                      f"{r['Delta_rad_s']/Nanowire.TWO_PI/1e9:>+12.4f} {r['g_over_Delta']:>+8.4f} "
                      f"{r['J_rad_s']/Nanowire.TWO_PI/1e3:>+16.3f}")
            print(f"Total J_1N/2pi = {self.J_1N_cyclic/1e6:.4f} MHz "
                  f"({self.J_1N/1e6:.3f} Mrad/s); iSWAP time = {self.iswap_time()*1e9:.1f} ns")
        return rows
    
    def plot_phonon_modes(self, xrange=None, savefig=False):
        units = 1e-9 / Nanowire.TWO_PI   # rad/s -> GHz (cyclic)
        fig, ax = plt.subplots(figsize=(12, 3))
        for freq in self.eigenfrequencies:
            ax.vlines(freq * units, 0, 1, colors="black")
        ax.vlines(self.omega_0[0]*units, 0, 1, colors="red")
        plt.tick_params(left = False, labelleft = False)
        ax.set_xlabel("Mode frequency $\\omega_m/2\\pi$ (GHz)", fontsize=fontsize_axis)
        if xrange:
            ax.set_xlim(xrange[0]*units, xrange[1]*units)
        if savefig:
            plt.savefig(
                f"plots/nanowires/plot_phonon_modes_n={self.n_electrons}.pdf", bbox_inches='tight', dpi=1000
            )
        plt.show()

    def plot_phonon_eigenmodes(self, savefig=False):
        n_cols = 3  # You can adjust this
        n_rows = int(np.ceil((self.n_electrons-2) / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 3 * n_rows),
            sharex=True, sharey=True,
            squeeze=False
        )

        for i in range(self.n_electrons-2):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row][col]

            y = self.eigenmodes[:, 2, i]
            x = range(1,len(y)+1)

            # Plot low-opacity line
            ax.plot(
                x, y,
                linestyle='-',
                color='blue',
                alpha=0.18,
                linewidth=1.0
            )

            # Plot opaque blue markers
            ax.plot(
                x, y,
                linestyle='',
                marker='o',
                color='blue',
                alpha=1.0,
                markersize=6,
                markeredgewidth=0.5
            )

            # Label first and last points
            ax.text(x[0], y[0], f"{y[0]:.3f}", fontsize=fontsize_legend, va='bottom', ha='left')
            ax.text(x[-1], y[-1], f"{y[-1]:.3f}", fontsize=fontsize_legend, va='bottom', ha='right')

            ax.set_title(f"Mode {i+1}", fontsize=fontsize_legend)
            if row == n_rows - 1:
                ax.set_xlabel("Electron", fontsize=fontsize_axis)
                ax.set_xticks(range(1, self.n_electrons + 1))

        # Remove unused subplots
        for j in range(self.n_electrons, n_rows * n_cols):
            fig.delaxes(axes[j // n_cols][j % n_cols])

        # Shared y-axis label
        fig.text(0.04, 0.5, "Phonon $y$ amplitude", va='center', rotation='vertical', fontsize=fontsize_axis)

        fig.tight_layout(rect=[0.07, 0, 1, 1])  # Leave space for y-label
        if savefig:
            plt.savefig(
                f"plots/nanowires/plot_phonon_eigenmodes_n={self.n_electrons}.pdf", bbox_inches='tight', dpi=1000
            )
        plt.show()

    def plot_interactions(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        pos = ax.imshow(self.interactions, cmap='Blues', interpolation='none')
        fig.colorbar(pos, ax=ax)
        plt.show()

    def plot_nanowire_positions(self, savefig=False, plot_potentials=False):
        # fig = plt.figure(figsize=[12, 12])
        fig = plt.figure(figsize=[10, 8])
        ax = fig.add_subplot(111, projection="3d")
        max_position = 0
        for position in self.positions:
            # z, x, y = position[2] * Nanowire.l, position[0] * Nanowire.l, position[1] * Nanowire.l
            z, x, y = position[2] * 1e6, position[0] * 1e6, position[1] * 1e6
            ax.scatter(z, x, y, color="blue")
            if abs(z) > max_position:
                max_position = abs(z)
        if plot_potentials: 
            def fn_axial_potential(z):
                # potential1 = self.mass * self.dot_freqs[2] ** 2 * (z - (self.dot_centres[0] * Nanowire.l)) ** 2 / 2
                # potentialwire = self.mass *  self.trap_freqs[2] ** 2 * (z - (self.dot_centres[1] * Nanowire.l)) ** 2 / 2
                # potentialn =  self.mass * self.dot_freqs[2] ** 2 * (z - (self.dot_centres[2] * Nanowire.l)) ** 2 / 2
                potential1 = self.mass * self.dot_freqs[2] ** 2 * (z - (self.dot_centres[0] * 1e6)) ** 2 / 2
                potentialwire = self.mass *  self.trap_freqs[2] ** 2 * (z - (self.dot_centres[1] * 1e6)) ** 2 / 2
                potentialn =  self.mass * self.dot_freqs[2] ** 2 * (z - (self.dot_centres[2] * 1e6)) ** 2 / 2
                return min(potential1, potentialwire, potentialn)
            zs = np.linspace(-1.1*max_position, 1.1*max_position, 1000)
            xs = [0 for _ in zs]
            potentials = np.array([fn_axial_potential(z) for z in zs])
            rescaled_potentials = (max_position/2) * (potentials - min(potentials)) / (max(potentials) - min(potentials))
            ax.plot(zs, xs, rescaled_potentials, color="red")

        ticks = [-max_position, -max_position / 2, 0, max_position / 2, max_position]
        ax.set_xticks(ticks)
        ax.set_zticks([-2 * max_position, -max_position, 0, max_position, 2 * max_position])
        # ax.set_zticks()
        # ax.set_xlabel("Axial $y~(\\mathrm{\\mu m})$", fontsize=fontsize_axis)
        # ax.set_ylabel("$x~(\\mathrm{\\mu m})$", fontsize=fontsize_axis)
        # ax.set_zlabel("$z~(\\mathrm{\\mu m})$", fontsize=fontsize_axis)
        ax.set_xlabel("Axial $y~(\\mathrm{\\mu m})$", labelpad=10, fontsize=fontsize_axis)
        ax.set_ylabel("Transverse $x~(\\mathrm{\\mu m})$", labelpad=10, fontsize=fontsize_axis)
        ax.set_zlabel("$z~(\\mathrm{\\mu m})$", labelpad=10, fontsize=fontsize_axis)
        ax.tick_params(axis="both", labelsize=fontsize_ticks - 6)
        if savefig:
            plt.savefig(
                f"plots/nanowires/plot_nanowire_positions_n={self.n_electrons}.pdf"
            )
        plt.show()

    def plot_nanowire_positions_2d(self, savefig=False, plot_potentials=False):
        # fig = plt.figure(figsize=[12, 12])
        fig, ax = plt.subplots(figsize=(8, 5))
        max_position = 0
        len_scale = Nanowire.l * 1e6  # Convert length scale to micrometers
        for position in self.positions:
            z, x = position[2] * len_scale, position[0] * len_scale
            ax.scatter(z, x, color="blue")
            if abs(z) > max_position:
                max_position = abs(z)
        if plot_potentials: 
            def fn_axial_potential(z):
                potential1 = self.mass * self.dot_freqs[2] ** 2 * (z - (self.dot_centres[0] * len_scale)) ** 2 / 2
                potentialwire = self.mass *  self.trap_freqs[2] ** 2 * (z - (self.dot_centres[1] * len_scale)) ** 2 / 2
                potentialn =  self.mass * self.dot_freqs[2] ** 2 * (z - (self.dot_centres[2] * len_scale)) ** 2 / 2
                return min(potential1, potentialwire, potentialn)
            zs = np.linspace(-1.1*max_position, 1.1*max_position, 1000)
            potentials = np.array([fn_axial_potential(z) for z in zs])
            rescaled_potentials = (max_position/2) * (potentials - min(potentials)) / (max(potentials) - min(potentials))
            ax.plot(zs, rescaled_potentials, color="red")

        ticks = [-max_position, -max_position / 2, 0, max_position / 2, max_position]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        # ax.set_zticks(ticks)
        ax.set_xlabel("Axial $y~(\\mathrm{\\mu m})$", fontsize=fontsize_axis)
        ax.set_ylabel("$z~(\\mathrm{\\mu m})$", fontsize=fontsize_axis)
        # ax.set_zlabel("$y~(\\mathrm{\\mu m})$", fontsize=fontsize_axis)
        ax.tick_params(axis="both", labelsize=fontsize_ticks - 6)
        if savefig:
            plt.savefig(
                f"plots/nanowires/plot_nanowire_positions_2d_n={self.n_electrons}.pdf"
            )
        plt.show()

    def plot_spin_interactions(
        self, x=0, include_ion_chain=False, savefig=False, plot_label=None
    ):
        if x:
            electrons = [x - i for i in range(x)] + [
                i - x for i in range(x + 1, self.n_electrons)
            ]
            Js = np.delete(self.interactions[x], x)
        else:
            electrons = [i for i in range(1, self.n_electrons)]
            interactions = self.interactions_cyclic[x][1:]

        if plot_label:
            fig, ax = plt.subplots(figsize=[10, 10])
        else:
            fig, ax = plt.subplots(figsize=figure_size)

        linestyles = ["", "dashed", "dashed"]
        markers = ["d", "", ""]
        colours = ["b", "r", "g"]

        # ys = [Js, alpha_fit]
        # legend = [f"Numerical $J_{{1 j}}$", f"${popt[0]:.2f} r^{{-{popt[1]:.3f}}}$"]
        legend = [f"Numerical $J_{{1 j}}$"]

        ax.plot(
            electrons, interactions, linestyle=linestyles[0], color=colours[0], marker=markers[0]
        )

        ax.legend(legend, fontsize=fontsize_legend)

        ax.set_xlabel(xlabel="$r = j - 1$", fontsize=fontsize_axis)
        ax.set_ylabel(ylabel="Coupling strength $J_{1 j}/2\\pi$ (Hz)", fontsize=fontsize_axis)

        ax.grid()
        # ax.set(xscale="log")
        # ax.set(yscale="log")
        plt.xticks(fontsize=fontsize_ticks)
        plt.yticks(fontsize=fontsize_ticks)

        if plot_label:
            ax.text(
                0.025, 0.90, plot_label, fontsize=22, transform=plt.gcf().transFigure
            )

        if include_ion_chain:
            ax2 = fig.add_axes([0.48, 0.38, 0.45, 0.42], projection="3d")
            max_position = 0
            l = 1e6
            for position in self.positions:
                x, y, z = position[2] * l, position[0] * l, position[1] * l
                ax2.scatter(x, y, z, color="blue")
                if abs(x) > max_position:
                    max_position = abs(x)
            ticks = [
                -max_position,
                -max_position / 2,
                0,
                max_position / 2,
                max_position,
            ]
            ax2.set_xticks(ticks)
            ax2.set_yticks(ticks)
            ax2.set_zticks(ticks)
            ax2.set_xlabel("Axial $z~(\\mathrm{\\mu m})$", fontsize=fontsize_axis - 4)
            ax2.set_ylabel("$x~(\\mathrm{\\mu m})$", fontsize=fontsize_axis - 4)
            ax2.set_zlabel("$y~(\\mathrm{\\mu m})$", fontsize=fontsize_axis - 4)
            ax2.tick_params(axis="both", labelsize=fontsize_ticks - 8)

        plt.tight_layout()
        if savefig:
            plt.savefig(
                f"plots/interactions/plot_electron_positions_n={self.n_electrons}.pdf"
            )
        plt.show()
