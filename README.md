# Bayesian-Flood-depth-damage-
Bayesian updating for generic curves to state specific

Hierarchical Bayesian Depth-Damage Curve Modeling

Project Overview

This Python script implements a hierarchical Bayesian model to estimate state-specific depth-damage curves for flood risk assessment. It uses the Army Corps of Engineers (ACOE) generic curve data as a prior (population-level information) and combines it with synthetic, state-specific observations to produce more robust and localized damage curves.

The core of the model is built using the pymc library for probabilistic programming, leveraging a Beta likelihood function to model the damage fraction (which must be between 0 and 1).

Key Features

Hierarchical Structure: Shares strength across states, preventing poorly constrained curves in states with few observations.

Parametric Form: Uses the common $D(d) = 1 - e^{-\alpha d^\beta}$ curve form, where $D(d)$ is the damage fraction at depth $d$.

Synthetic Data Generation: Includes detailed synthetic data generation for both the generic ACOE table and state-level observations, simulating realistic noise and underreporting issues.

Diagnostic Outputs: Generates trace plots, summary statistics, and posterior predictive checks using arviz.

Prerequisites

This script requires the following Python libraries:

pip install numpy pandas matplotlib scipy pymc arviz pytensor


Note: This code is written for PyMC v4+ and utilizes ArviZ for diagnostics.




The script will run the Markov Chain Monte Carlo (MCMC) sampler and automatically generate several plots, including the synthetic data, MCMC trace plots, posterior predictive checks, and final state-specific curves.

Script Breakdown (10 Sections)

The script is structured into 10 logical sections for clarity and reproducibility:

Section

Title

Description

1 & 2

Define Parametric Form & Synthetic ACOE Table

Defines the S-curve function $D(d)$ and generates a synthetic dataset representing the generic, tabular depth-damage curve often provided by federal agencies (ACOE).

3

Synthetic State Observations

Generates the primary observational data. This simulates damage reports from multiple states (n_states=8), each with slightly different true underlying parameters ($\alpha$ and $\beta$) and realistic noise (heteroskedasticity and underreporting).

4

Fit Army Corps Curve to Get Priors

Performs a simple non-linear least squares fit (scipy.optimize.curve_fit) on the generic ACOE data. The mean and standard deviation of these fitted parameters are used to inform the hyperpriors in the Bayesian model.

5

Build Hierarchical PyMC Model

Defines the core Bayesian model:<ul><li>Hyperpriors: Normal priors centered on the ACOE fitted values ($\mu_{\alpha}$, $\mu_{\beta}$) govern the population-level means.</li><li>State-Level Parameters: $\alpha_{s}$ and $\beta_{s}$ are modeled as draws from the population distribution, allowing them to vary by state ID (state_ids).</li><li>Likelihood: A Beta distribution is used for the likelihood of the damage fraction (y_obs), as it naturally constrains the output between (0, 1). A dispersion parameter ($\kappa$) controls the precision.</li></ul>The model then runs the MCMC sampling to generate the trace (posterior samples).

6

Posterior Diagnostics & Summary

Plots the MCMC trace and prints the summary statistics (mean, SD, R-hat, ESS) for the hyperprior parameters, ensuring the model converged correctly.

7

Posterior-Predictive Checks (PPC)

Generates samples from the posterior predictive distribution and plots them against the original observed data. This is a crucial step for checking model fit and identifying potential systematic errors or outliers.

8

Compute Updated Curves per State

Extracts the posterior samples for the state-level parameters ($\alpha_s$, $\beta_s$) and uses them to compute a full range of damage fractions across a fine depth_grid. It calculates the mean curve and the 95% Credible Interval (Lo/Hi) for each state.

9

Plot State Curves vs. Army Generic

Visualizes the final, updated damage curves for the first six states. Each plot compares: 1) The state's observed data, 2) The Bayesian posterior mean and 95% CI, and 3) The original ACOE generic fitted curve, highlighting the effect of the local data on the generic prior.



Data Clipping: All damage fractions are clipped to be strictly within $(10^{-6}, 1 - 10^{-6})$. This is MANDATORY because the Beta distribution is defined only on the open interval (0, 1).

Safe Computation: pytensor.tensor functions (pt.power, pt.clip) are used to ensure numerically stable computations within the PyMC graph, preventing overflow errors.
