Code for fitting selective prediction set models based on the paper: Jean Feng, Arjun Sondhi, Jessica Perry, and Noah Simon. "Selective prediction-set models with coverage rate guarantees" Under review.

Code is also included for reproducing simulation results and empirical analyses.


# Installation
We use `pip` to install things into a python virtual environment.

We use nestly + SCons to run simulations/analyses.
Then activate the virtual environment and then run `scons ___`.

# How to get things running quickly and how to run analyses in the paper
The full pipelines are specified in the sconscript files.
To run it, we use `scons`. For example: `scons simulation_compare/`
The sconscript files provides the full pipeline for generating data and analyzing it.
