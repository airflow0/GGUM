# GGUM

Generalized Graded Unfolding Model

1. Introduction
2. Formal Probability Equation of the GGUM
3. Goals
4. Methodology ( Needle in a haystack )
5. MatPlot Library Data
 
Generalized Graded Unfolding Model

The GGUM is an advanced item response theory model that incorporates the parameters of Alpha, Delta, Theta, and Tau parameters to estimate the probability of a person's attitude towards a certain question.

This program computes the Alpha, Delta, and Theta parameters of the GGUM. 


ggum_prob Function:
Algorithmic equivalent to the GGUM probability function. 

joint_log_prior Function:
Prior distributions that were necessary for the Bayesian inference using Hamilton Monte Carlo Markov Chain method.

log_likelihood:
The algorithmic likelihood function for the GGUM

joint_log_prob:
Combination of both joint_log_prior function and likelihood with correlation to the probability function.

Hamilton Monte Carlo Settings
5000 Steps
750 Burn-In Steps
Step_Size: 0.001

