import numpy as np
from math import sqrt, pi, prod, log, exp
from matplotlib import pyplot as plt

n=5000
num_params = 4
sigmoid = lambda x: np.exp(x) / (1 + np.exp(x))
sigmoid = lambda x: 1 / (1 + np.exp(-x))

rng = np.random.default_rng()

true_beta = np.array([0.8, -1.5, 0.6, 0.9])
x = rng.normal(size=(n, num_params))
#x[:,0] = 1
p = sigmoid(np.dot(x, true_beta))
y = p > rng.random(n)#rng.binomial(1, p)

prior_sd = 10
#prior = lambda beta: prod(np.exp(-((beta/prior_sd)**2)/2)/(prior_sd*sqrt(2*pi)))
log_prior = lambda beta: -sum(beta**2)/(2*prior_sd**2) - num_params*log(prior_sd*sqrt(2*pi))
grad_log_prior = lambda beta: - beta / prior_sd**2
init_beta = rng.normal(scale=prior_sd, size=num_params)


def propose(beta, M, M_inv):
	L=10
	epsilon=0.05

	if M is None:
		momentum = rng.normal(size=num_params) + grad_log_post(beta) * epsilon/2
		M_inv = np.identity(num_params)
	else:
		momentum = rng.multivariate_normal(np.zeros(num_params), M) + grad_log_post(beta) * epsilon/2

	for _ in range(L-1):
		beta += epsilon * np.dot(M_inv, momentum)
		momentum += grad_log_post(beta) * epsilon

	return beta + epsilon * np.dot(M_inv, momentum)

#propose = lambda beta: rng.normal(loc=beta, scale = sqrt(0.5))

#likelihood = lambda x_dot_beta: prod( np.power(sigmoid(x_dot_beta), y) * np.power(1/(1+np.exp(x_dot_beta)) , 1-y) )#sigmoid(- x_dot_beta), 1-y) )
#likelihood = lambda x_dot_beta: prod( np.power(sigmoid(x_dot_beta), y) * np.power(sigmoid(- x_dot_beta), 1-y) )
log_likelihood = lambda x_dot_beta: np.dot(y, x_dot_beta) - sum(np.log(1 + np.exp(x_dot_beta)))

#posterior = lambda beta: likelihood(np.dot(x, beta)) * prior(beta)
log_posterior = lambda beta: log_likelihood(np.dot(x, beta)) + log_prior(beta)

grad_log_like = lambda beta: np.dot(y, x) - np.sum(x/(1+np.exp(-np.dot(x, beta))).reshape(-1,1), axis=0)
grad_log_post = lambda beta: grad_log_like(beta) + grad_log_prior(beta)

#diff_of_ll_nums = lambda diff_of_x_dot_betas: np.dot(y, diff_of_x_dot_betas)
#ll_denom = lambda x_dot_beta: sum(np.log(1 + np.exp(x_dot_beta)))
#rest_of_log_posterior = lambda beta, x_dot_beta: log_prior(beta) - ll_denom(x_dot_beta)


# def diff_of_log_posts(prop_beta, beta):
# 	x_dot_prop_beta = np.dot(x, prop_beta)
# 	x_dot_beta = np.dot(x, beta)
# 	return diff_of_ll_nums(x_dot_prop_beta-x_dot_beta) + rest_of_log_posterior(prop_beta, x_dot_prop_beta) - rest_of_log_posterior(beta, x_dot_beta)
# 	#return log_posterior(prop_beta) - log_posterior(beta)

def mh(beta, curr_lp, M=None, M_inv=None):
	prop_beta = propose(beta, M, M_inv)
	prop_lp = log_posterior(prop_beta)
	diff_lp = prop_lp - curr_lp#diff_of_log_posts(prop_beta, beta)
	if diff_lp > 0 or rng.random() < exp(diff_lp):
	#if rng.random() < min(1, posterior(prop_beta)/posterior(beta)):
		beta = prop_beta
		curr_lp = prop_lp
	return beta, curr_lp

def calc_post(init_beta, label="", color='b', seed=100, reg=1e-42):
	n_iter = 1000
	n_iter = int(n_iter/4)

	rng = np.random.default_rng(seed)
	beta = init_beta
	new_metric = None
	covariance = None


	for j in range(3):
		if j > 1:
			n_iter *= 2
			#1 1 2 4
		curr_lp = log_posterior(beta)
		mean_beta = np.zeros(num_params)
		betas = np.zeros((n_iter, num_params))
		for i in range(n_iter):
			old_beta = beta
			beta, curr_lp = mh(beta, curr_lp, new_metric, covariance)
			mean_beta += beta
			betas[i] = beta

		mean_beta /= n_iter
		#sample_sd = np.sqrt(np.sum((betas - mean_beta)**2, axis=0)/n_iter)
		covariance = np.zeros((num_params, num_params))
		for beta in betas:
			covariance += np.outer(beta - mean_beta, beta - mean_beta)
		covariance /= n_iter
		covariance += reg * np.identity(num_params)
		print(mean_beta)
		print(covariance)
		new_metric = np.linalg.inv(covariance)
	n_iter *= 2
	# 8
	n_iter *= 2
	accepted = 0
	mean_beta = np.zeros(num_params)
	betas = np.zeros((n_iter, num_params))
	for i in range(n_iter):
		old_beta = beta
		beta, curr_lp = mh(beta, curr_lp, new_metric, covariance)
		if (beta != old_beta).any():
			accepted += 1
		mean_beta += beta
		betas[i] = beta

	mean_beta /= n_iter
	sample_sd = np.sqrt(np.sum((betas - mean_beta)**2, axis=0)/n_iter)
	#covariance = np.sum((np.outer(beta - mean_beta, beta - mean_beta) for beta in betas), axis=0)/n_iter

	for i in range(num_params):
		plt.figure(i)
		plt.plot(betas[:,i], label=label)
		plt.axhline(mean_beta[i], color=color)
		plt.fill_between(range(-int(n_iter/20), n_iter+int(n_iter/20)), mean_beta[i]-sample_sd[i], mean_beta[i]+sample_sd[i], alpha=.2)

	return accepted/n_iter, mean_beta, sample_sd#, covariance

num_walks = 10
h_acpts = np.zeros(num_walks)
h_means = np.zeros((num_params, num_walks))

for i in range(num_walks):
	rng = np.random.default_rng(i)
	# x = rng.normal(size=(n, num_params))
	# x[:,1] = 1
	# p = sigmoid(np.dot(x, true_beta))
	# y = p > rng.random(n)#rng.binomial(1, p)

	a, b, _ = calc_post(init_beta)
	h_acpts[i] = a
	h_means[:, i] = b



propose = lambda beta, _, __: rng.normal(loc=beta, scale = sqrt(0.5))
g_acpts = np.zeros(num_walks)
g_means = np.zeros((num_params, num_walks))

for i in range(num_walks):
	rng = np.random.default_rng(i)
	# x = rng.normal(size=(n, num_params))
	# x[:,1] = 1
	# p = sigmoid(np.dot(x, true_beta))
	# y = p > rng.random(n)#rng.binomial(1, p)

	a, b, _ = calc_post(init_beta)
	g_acpts[i] = a
	g_means[:, i] = b

for i in range(num_params):
	plt.figure(i)
	plt.clf()
	plt.plot(h_means[i], h_acpts, label="Hamiltonian MC", marker='.', linestyle='')
	plt.plot(g_means[i], g_acpts, label="Gaussian MH", marker='.', linestyle='')

	plt.axvline(true_beta[i], color='m', label="true value")
	plt.xlabel("parameter")
	plt.ylabel("acceptance rate")
	plt.legend()

print(sum(np.abs(h_means[:,i] - true_beta) for i in range(num_walks))/num_walks)
#[0.03124158 0.01676994 0.03185821 0.01922252]

# label = "Hamiltonian MC"
# print(calc_post(init_beta, label, 'b'))
# label = "Gaussian proposal"
# propose = lambda beta: rng.normal(loc=beta, scale = sqrt(0.5))
# print(calc_post(init_beta, label, 'orange'))

# for i in range(num_params):
# 	plt.figure(i)
# 	plt.axhline(true_beta[i], label="true value", color='m')
# 	plt.legend()
plt.show()
