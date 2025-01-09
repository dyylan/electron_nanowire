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

    k = 8.98755179e9 / 12.9  # coulomb constant / relative permittivity in GaAs 
    hbar = 1.054571817e-34  # reduced planck constant
    e = 1.602176634e-19  # charge of electron
    m_e = 9.1093837e-31 # electron mass
    l = 1 # length scale (was 1e-6 before)
    ge = 0.39 # electron g-factor in GaAs quantum dot
    mub = 9.2740100783e-24 # Bohr Magneton
    c = 299792458 # speed of light

    def __init__(self, 
        n_electrons, 
        # trap_freqs = 2 * np.pi * np.array([4e12, 15e12, 0.15e12]),
        # dot_freqs = 2 * np.pi * np.array([4e12, 15e12, 4e12]),
        # trap_freqs = np.array([4e12, 15e12, (4/10)*1e12]),
        # trap_freqs = np.array([1e12, 15e12, (1/10)*1e12]),
        # trap_freqs = np.array([0.5e12, 15e12, (1/10)*0.5e12]),
        trap_freqs = np.array([0.85e12, 15e12, (1/10)*0.85e12]),
        # trap_freqs = np.array([0.35e12, 15e12, (1/10)*0.35e12]),
        # dot_freqs = np.array([4e12, 15e12, 4e12]),
        # dot_freqs = np.array([1e12, 15e12, 1e12]),
        # dot_freqs = np.array([0.5e12, 15e12, 0.5e12]),
        dot_freqs = np.array([0.85e12, 15e12, 0.85e12]),
        # dot_freqs = np.array([0.35e12, 15e12, 0.35e12]),
        # trap_freqs = np.array([0.85e12, 15e12, (0.85/10)*1e12]),
        # dot_freqs = np.array([0.85e12, 15e12, 0.85e12]),
        # dot_centres = np.array([-350e-9, 0, 350e-9]),
        # dot_centres = np.array([-350e-9, 0, 350e-9]),
        # dot_centres = np.array([-1050e-9, 0, 1050e-9]),
        # dot_centres = np.array([-700e-9, 0, 700e-9]),
        dot_centres = np.array([-1000e-9, 0, 1000e-9]),
        # dot_centres = [-1e-10, 0, 1e10],
        effective_mass=0.067,
        # effective_mass=0.041,
        # omega_0 = np.ones(1) * (39738353063.18441 + 1e3),
        # omega_0 = np.ones(1) * (1.9869176531591518e9 + 1e5),
        # omega_0 = np.ones(1) * (2.515e13),
        # omega_0 = np.ones(1) * (2.51327412e+13 + 1e10),
        # omega_0 = np.ones(1) * (4e+12 + 1e12),
        omega_0 = np.ones(1) * (1.75e+11),
        # omega_0 = np.ones(1) * (3.6e+11),
        # alpha_R = 0.1e-10, # Rashba term in units of eV.m 
        alpha_R = 0.3e-10
        ):
        self.n_electrons = n_electrons
        self.trap_freqs = trap_freqs
        self.dot_freqs = dot_freqs
        self.dot_centres = dot_centres
        self.effective_mass = effective_mass
        self.mass = effective_mass * Nanowire.m_e
        self.omega_0 = omega_0
        # self.alpha_R = np.reshape(((-1 * Nanowire.ge * Nanowire.mub * electric_field / (2 * self.mass * Nanowire.c))
        #     * np.array([1] + [0 for _ in range(n_electrons - 2)] + [1])), (self.n_electrons, 1)) / Nanowire.hbar
        # print(f"alpha_R = {self.alpha_R}")
        self.alpha_R = (alpha_R * Nanowire.e / Nanowire.hbar) * np.reshape(np.array([1] + [0 for _ in range(n_electrons - 2)] + [1]), (self.n_electrons, 1))
        # self.alpha_R = (alpha_R * Nanowire.e) * np.reshape(np.array([1] + [0 for _ in range(n_electrons - 2)] + [1]), (self.n_electrons, 1))
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
                options={"maxiter": 1000, "disp": True},
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
                options={"maxiter": 1000, "disp": True},
                tol=1e-15,
            )

        def op_val(op):
            return Nanowire.expr_tot(op.x, self.n_electrons, self.mass, self.trap_freqs, self.dot_freqs, self.dot_centres, Nanowire.l)

        if op_val(op_powell) < op_val(op_cg):
            op = op_powell
            print("Using Powell optimizer result.")
        else:
            op = op_cg
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
            np.array([1,0,0]),
            # np.sqrt(Nanowire.hbar * self.mass * self.eigenfrequencies / 2 ),
            np.sqrt(self.mass * self.eigenfrequencies / 2 ),
        )
        g = np.einsum("im,in->imn", self.alpha_R, eta)
        Js = np.einsum(
            "ij,imn,jmn,n,mn->ij",
            1 - np.identity(self.n_electrons),
            g,
            g,
            self.eigenfrequencies,
            1 / np.subtract.outer(self.omega_0 ** 2, self.eigenfrequencies ** 2),
        )
        ratios = np.einsum(
            "imn,mn->in",
            g/np.sqrt(Nanowire.hbar),
            1/(np.subtract.outer(self.omega_0, self.eigenfrequencies))
        )
        detunings = [abs(self.omega_0[0] - freq) for freq in self.eigenfrequencies]
        # print(f"maximum g = {max(g.flatten()) / np.sqrt(Nanowire.hbar)}")
        # print(f"detuning = {min(detunings) }")
        # print(f"g/detuning = {(max(g.flatten()) / np.sqrt(Nanowire.hbar)) / min(detunings)}")
        # print(f"g/detuning = {(max(g.flatten()) / np.sqrt(Nanowire.hbar)) / 1e12}")
        print(f"maximum g/detuning = {max(abs(ratios[0]))}")
        return Js / Nanowire.hbar 

    def plot_phonon_modes(self, xrange=None, savefig=False):
        units = 1e-12
        fig, ax = plt.subplots(figsize=(12, 3))
        for freq in self.eigenfrequencies:
            ax.vlines(freq * units, 0, 1, colors="black")
        ax.vlines(self.omega_0[0]*units, 0, 1, colors="red")
        plt.tick_params(left = False, labelleft = False)
        ax.set_xlabel("Phonon modes (THz)", fontsize=fontsize_axis)
        if xrange:
            ax.set_xlim(xrange[0]*units, xrange[1]*units)
        if savefig:
            plt.savefig(
                f"plots/nanowires/plot_phonon_modes_n={self.n_electrons}.png", bbox_inches='tight', dpi=1000
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
        fig, ax = plt.subplots(figsize=(10, 8))
        max_position = 0
        for position in self.positions:
            z, x = position[2] * Nanowire.l, position[0] * Nanowire.l
            ax.scatter(z, x, color="blue")
            if abs(z) > max_position:
                max_position = abs(z)
        if plot_potentials: 
            def fn_axial_potential(z):
                potential1 = self.mass * self.dot_freqs[2] ** 2 * (z - (self.dot_centres[0] * Nanowire.l)) ** 2 / 2
                potentialwire = self.mass *  self.trap_freqs[2] ** 2 * (z - (self.dot_centres[1] * Nanowire.l)) ** 2 / 2
                potentialn =  self.mass * self.dot_freqs[2] ** 2 * (z - (self.dot_centres[2] * Nanowire.l)) ** 2 / 2
                return min(potential1, potentialwire, potentialn)
            zs = np.linspace(-1.1*max_position, 1.1*max_position, 1000)
            potentials = np.array([fn_axial_potential(z) for z in zs])
            rescaled_potentials = (max_position/2) * (potentials - min(potentials)) / (max(potentials) - min(potentials))
            ax.plot(zs, rescaled_potentials, color="red")

        ticks = [-max_position, -max_position / 2, 0, max_position / 2, max_position]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        # ax.set_zticks(ticks)
        ax.set_xlabel("Axial $z~(\\mathrm{\\mu m})$", fontsize=fontsize_axis)
        ax.set_ylabel("$x~(\\mathrm{\\mu m})$", fontsize=fontsize_axis)
        # ax.set_zlabel("$y~(\\mathrm{\\mu m})$", fontsize=fontsize_axis)
        ax.tick_params(axis="both", labelsize=fontsize_ticks - 6)
        if savefig:
            plt.savefig(
                f"plots/nanowires/plot_nanowire_positions_n={self.n_electrons}.pdf"
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
            interactions = self.interactions[x][1:]

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
        ax.set_ylabel(ylabel="Coupling strength $J_{1 j}$ (Hz)", fontsize=fontsize_axis)

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
