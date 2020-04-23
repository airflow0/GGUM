import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sys
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)
tfd = tfp.distributions
sess = tf.Session()

def evaluate(tensors):
    return sess.run(tensors)

def load_data():
    t_responses = []
    with open('data/data.txt') as file:
        for line in file:
            temp_ = line.rsplit()
            temp_responses_characterized = []
            for char in temp_[1]:
                temp_responses_characterized.append(float(char))
            t_responses.append(temp_responses_characterized)
    return t_responses

responses = load_data()
responses_numpify = np.array(responses, dtype=np.float64)
N = responses_numpify.shape[0]
I = responses_numpify.shape[1]

responses_numpify = np.delete(responses_numpify,slice(50,57) , axis=1)
N = responses_numpify.shape[0]
I = responses_numpify.shape[1]

print(responses_numpify)
K = 5
k = K-1

data = tf.cast(responses_numpify, tf.int32)

@tf.function
def ggum_prob(theta, alpha, delta, expanded=False):
    # logit_left = tf.exp(tf.multiply(alpha[tf.newaxis, :, tf.newaxis],tf.subtract(theta[:, tf.newaxis, tf.newaxis], delta[tf.newaxis, :, :])))
    # logit_right = tf.exp(tf.multiply(alpha[tf.newaxis, :, tf.newaxis], tf.multiply(float(2 * K - 1 - k), tf.subtract(theta[:, tf.newaxis, tf.newaxis], delta[tf.newaxis, :, :]))))

    theta_ = theta[:, tf.newaxis]
    alpha_ = alpha[tf.newaxis, :]
    delta_ = delta[tf.newaxis, :]

    logit_left = tf.exp( alpha_ * (theta_ - delta_))
    logit_right = tf.exp(alpha_ * (2 * K-1 - k) * (theta_ - delta_) )
    numerator = tf.add(logit_left, logit_right)
    denominator = tf.reduce_sum(tf.add(logit_left, logit_right))
    probs = tf.divide(numerator,denominator)
    return probs


@tf.function
def joint_log_prob(responses, alpha, delta, theta):
    return tf.reduce_sum(log_likelihood(responses, alpha, delta, theta)) +\
           joint_log_prior(alpha, delta, theta)


@tf.function
def log_likelihood(responses, alpha, delta, theta):
    rv_responses = tfd.Bernoulli(probs=ggum_prob(theta, alpha, delta))
    return rv_responses.log_prob(responses)


@tf.function
def joint_log_prior(alpha, delta, theta):
    rv_alpha = tfd.Normal(loc=0., scale=1.)
    rv_delta = tfd.Normal(loc=0.0, scale=4.0)
    rv_theta = tfd.Normal(loc=0.0, scale=1.)

    return tf.reduce_sum(rv_alpha.log_prob(alpha)) + \
           tf.reduce_sum(rv_delta.log_prob(delta)) + \
           tf.reduce_sum(rv_theta.log_prob(theta))




initial_chain_state = [
    tf.ones((I), name='init_alpha'),
    tf.ones((I), name='init_delta'),
    tf.zeros((N), name='init_theta'),

]
unconstraining_bijectors = [
    tfp.bijectors.SoftmaxCentered(),  # R^+ \to R
    tfp.bijectors.Identity(),  # Maps R to R.
    tfp.bijectors.Identity(),  # Maps R to R.
]

unnormalized_posterior_log_prob = tf.function(lambda *args: joint_log_prob(data, *args))
number_of_steps= 5000
burnin=750
num_leapfrog_steps=2
hmc=tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=num_leapfrog_steps,
        step_size=0.001,
        state_gradients_are_stopped=True),
    bijector=unconstraining_bijectors)

kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel=hmc, num_adaptation_steps=int(burnin * 0.8))

def trace_everything(states, previous_kernel_results):
  return previous_kernel_results

[alpha,delta,theta], kernel_results = tfp.mcmc.sample_chain(
    num_results=number_of_steps,
    num_burnin_steps=burnin,
    current_state=initial_chain_state,
    trace_fn=trace_everything,
    kernel=kernel)

with tf.device('/device:GPU:0'):
    [
        alpha_,
        delta_,
        theta_,
        kernel_results_
    ] = evaluate(
        [alpha,
         delta,
         theta,
         kernel_results
         ])
print('[THETAS]')
theta_mean = np.mean(theta_[burnin:],axis=0)
print(theta_mean)
print('[THETAS STD]')
print(np.std(theta_[burnin:],axis=0))
print('[DELTAS]')
delta_mean = np.mean(delta_[burnin:],axis=0)
print(delta_mean)
print('[ALPHAS]')
print(np.mean(alpha_[burnin:],axis=0))



plt.plot(theta_[burnin:, 0])
plt.show()

plt.plot(theta_mean)
plt.show()