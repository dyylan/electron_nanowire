import numpy as np
import matplotlib.pyplot as plt
import nanowire

from scipy.stats import norm

import scienceplots
plt.style.use('science')
fontsize_title=22
fontsize_label=24
fontsize_legend=16
fontsize_ticks=20

def plot_Js_gaussian_distribution(Js, savefig=False):
    Js = np.array(Js) * 1e-6  # Convert to MHz
    mu, sigma = norm.fit(Js)
    print(f"Fitted Gaussian parameters: mu = {mu}, sigma = {sigma}")

    # 3. Plot histogram
    plt.figure(figsize=(8, 4))
    count, bins, _ = plt.hist(
        Js,
        bins=30,
        density=False,
        alpha=0.4,
        color="tab:green",
        edgecolor="black",
        label="Number of samples"
    )

    # 4. Gaussian curve
    x = np.linspace(bins[0], bins[-1], 1000)
    pdf = norm.pdf(x, mu, sigma)
    plt.plot(x, pdf * np.diff(bins)[0] * len(Js), linewidth=2, color="tab:red", label=f"Gaussian fit:\n$\mu={mu:.3f}, \sigma={sigma:.3f}$")

    # 5. Plot data points over distribution (rug plot)
    plt.scatter(
        Js,
        np.zeros_like(Js)+0.01,  # slightly below axis
        marker="|",
        s=300,
        color="black",
        alpha=0.9,
        label="Sample J values"
    )
    from matplotlib.ticker import MaxNLocator
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.title("Interactions strengths with Gaussian Fit")
    plt.xlabel("$J_{1 N} $ (MHz)")
    # plt.ylabel("Samples count")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    if savefig:
        plt.savefig("plots/Js_from_noisy_trap_freq.pdf", dpi=300)
    plt.show()


if __name__=="__main__":
    samples = 10000
    trapping_freq = 0.6e12
    error_ratio = 0.01

    Js = []

    delta_x_trap_freq = np.random.normal(0, trapping_freq*error_ratio, samples)
    delta_y_trap_freq = np.random.normal(0, trapping_freq*error_ratio, samples)
    delta_x_dot_freq = np.random.normal(0, trapping_freq*error_ratio, samples)
    delta_y_dot_freq = np.random.normal(0, trapping_freq*error_ratio, samples)
    
    for i in range(samples):

        trap_freqs = np.array([trapping_freq + delta_x_trap_freq[i], 
                               15e12, 
                               (1/10)*(trapping_freq + delta_y_trap_freq[i])])
        
        dot_freqs = np.array([trapping_freq + delta_x_dot_freq[i], 
                              15e12, 
                              trapping_freq + delta_y_dot_freq[i]])
        
        wire = nanowire.Nanowire(n_electrons=8, trap_freqs=trap_freqs, dot_freqs=dot_freqs, verbose=False)
    
        Js.append(max(abs(wire.interactions[0])))
        print(f"[{i+1}/{samples}] J = {Js[-1]}.                   ", end="\r")

    plot_Js_gaussian_distribution(Js, savefig=True)

    