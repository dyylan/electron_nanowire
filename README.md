# Electron nanowire
Simulate the phonon bus of an electron nanowire. Generate the results for the paper: https://arxiv.org/abs/2603.13519

## Units
All frequency-like quantities inside `nanowire.Nanowire` (trap frequencies, mode
frequencies, `omega_0`, `g`, `J`, detunings) are **angular** frequencies in rad/s.
Use the `*_cyclic` properties or divide by `2*pi` to quote ordinary frequencies in
MHz/GHz. The effective spin-orbit coefficient used for all results is
`alpha_R = 2e-12 eV m` (alpha + beta for a gated GaAs well with the chain along [110]).

## Reproducing the paper
```
pip install autograd numpy scipy matplotlib
python3 reproduce_paper.py            # N = 8 headline numbers + mode-resolved table
python3 reproduce_paper.py 1e-12      # same with a more conservative alpha_eff
```
