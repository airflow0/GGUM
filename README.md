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



For a detailed process of the algorithms used, please visit: https://teams.microsoft.com/_?tenantId=45f26ee5-f134-439e-bc93-e6c7e33d61c2#/school/tab::6d287bc5-9f86-4b1f-96f8-1711c8c3cce4/General?threadId=19:a034b0d92b794ed6843e07f8a1641aab@thread.tacv2&ctx=channel
