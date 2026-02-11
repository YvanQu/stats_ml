import numpy as np
from math import sqrt, pi, prod, log, exp
from matplotlib import pyplot as plt


howlong = 3000
num_walks = 3

n=100
num_params = 4
sigmoid = lambda x: np.exp(x) / (1 + np.exp(x))
sigmoid = lambda x: 1 / (1 + np.exp(-x))

L=10
epsilon=0.05
cutoff=1000
rng = np.random.default_rng()

true_beta = np.array([0.8, -1.5, 0.6, 0.9])

x = rng.normal(size=(n, num_params))
x[:,0] = 1
p = sigmoid(np.dot(x, true_beta))
y = p > rng.random(n)#rng.binomial(1, p)

prior_sd = 10
#prior = lambda beta: prod(np.exp(-((beta/prior_sd)**2)/2)/(prior_sd*sqrt(2*pi)))
log_prior = lambda beta: -sum(beta**2)/(2*prior_sd**2) - num_params*log(prior_sd*sqrt(2*pi))
grad_log_prior = lambda beta: - beta / prior_sd**2
init_beta = rng.normal(scale=prior_sd, size=num_params)


def propose(beta):
	# L=20
	# epsilon=0.05

	momentum = rng.normal(scale=1, size=num_params) + grad_log_post(beta) * epsilon/2

	for _ in range(L-1):
		beta += epsilon * momentum
		momentum += grad_log_post(beta) * epsilon

	return beta + epsilon * momentum

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

def mh(beta, curr_lp):
	prop_beta = propose(beta)
	prop_lp = log_posterior(prop_beta)
	diff_lp = prop_lp - curr_lp#diff_of_log_posts(prop_beta, beta)
	if diff_lp > 0 or rng.random() < exp(diff_lp):
	#if rng.random() < min(1, posterior(prop_beta)/posterior(beta)):
		beta = prop_beta
		curr_lp = prop_lp
	return beta, curr_lp

def calc_post(init_beta, label="", color='b', seed=100):
	n_iter = int(howlong//3)
	rng = np.random.default_rng(seed)

	beta = init_beta
	betas = np.zeros((n_iter, num_params))
	all_betas = np.zeros((howlong, num_params))

	curr_lp = log_posterior(beta)
	for i in range(n_iter):
		beta, curr_lp = mh(beta, curr_lp)
		betas[i] = beta

	all_betas[:n_iter] = np.copy(betas)
	n_iter *= 2
	accepted = 0
	mean_beta = np.zeros(num_params)
	betas = np.zeros((n_iter, num_params))
	#beta = true_beta
	for i in range(n_iter):
		old_beta = beta
		beta, curr_lp = mh(beta, curr_lp)
		if (beta != old_beta).any():
			accepted += 1
		mean_beta += beta
		betas[i] = beta

	mean_beta /= n_iter
	sample_sd = np.sqrt(np.sum((betas - mean_beta)**2, axis=0)/n_iter)

	for i in range(num_params):
		plt.figure(i)
		plt.plot(betas[:,i], label=label)
		plt.axhline(mean_beta[i], color=color )
		plt.fill_between(range(-int(n_iter/20), n_iter+int(n_iter/20)), mean_beta[i]-sample_sd[i], mean_beta[i]+sample_sd[i], alpha=.2)

	fig, axes = plt.subplots(num_params, sharex=True)
	fig.supxlabel("iterations")
	fig.supylabel("parameter value")
	fig.suptitle(f"{accepted/n_iter} acceptance rate")

	all_betas[-n_iter:] = betas
	sd_beta = sample_sd

	for i in range(num_params):
		ax = axes[i]
		ax.set_ylim([mean_beta[i]-4*sd_beta[i], mean_beta[i]+4*sd_beta[i]])
		ax.plot(all_betas[:,i])#, label=label)
		ax.axhline(mean_beta[i])#, color=color)
		ax.axhline(true_beta[i], color='m')
		ax.fill_between(range(-int(howlong/20), howlong+int(howlong/20)), mean_beta[i]-sd_beta[i], mean_beta[i]+sd_beta[i], alpha=.2)

	return accepted/n_iter, mean_beta, sample_sd

h_acpts = np.zeros(num_walks)
h_means = np.zeros((num_params, num_walks))
h_errs = np.zeros((num_params, num_walks))
bunk_h = 0

for i in range(num_walks):
	rng = np.random.default_rng(i)
	# x = rng.normal(size=(n, num_params))
	# x[:,1] = 1
	# p = sigmoid(np.dot(x, true_beta))
	# y = p > rng.random(n)#rng.binomial(1, p)

	a, b, c = calc_post(init_beta)
	h_acpts[i] = a
	if (np.abs(b)<3).all():
		h_means[:, i] = b
		h_errs[:, i] = c
	else:
		h_means[:, i] = true_beta
		bunk_h += 1

plt.show()
plt.clf()

propose = lambda beta: rng.normal(loc=beta, scale = sqrt(0.05))
g_acpts = np.zeros(num_walks)
g_means = np.zeros((num_params, num_walks))
g_errs = np.zeros((num_params, num_walks))


for i in range(num_walks):
	rng = np.random.default_rng(i)
	# x = rng.normal(size=(n, num_params))
	# x[:,1] = 1
	# p = sigmoid(np.dot(x, true_beta))
	# y = p > rng.random(n)#rng.binomial(1, p)

	a, b, c = calc_post(init_beta)
	g_acpts[i] = a
	g_means[:, i] = b
	g_errs[:, i] = c

plt.show()
#########


def propose(beta, M, M_inv, adaptive_epsilon):
	# L=20
	# epsilon=0.02

	if M is None:
		momentum = rng.normal(size=num_params) + grad_log_post(beta) * epsilon/2
		M_inv = np.identity(num_params)
	else:
		try:
			momentum = rng.multivariate_normal(np.zeros(num_params), M) + grad_log_post(beta) * epsilon/2
		except:
			print(M)
			print(M_inv)
	for _ in range(L-1):
		beta += epsilon * np.dot(M_inv, momentum)
		momentum += grad_log_post(beta) * epsilon

	return beta + epsilon * np.dot(M_inv, momentum)


def mh(beta, curr_lp, adaptive_epsilon, M=None, M_inv=None):
	prop_beta = propose(beta, M, M_inv, adaptive_epsilon)
	prop_lp = log_posterior(prop_beta)
	diff_lp = prop_lp - curr_lp#diff_of_log_posts(prop_beta, beta)
	if diff_lp > 0 or rng.random() < exp(diff_lp):
	#if rng.random() < min(1, posterior(prop_beta)/posterior(beta)):
		beta = prop_beta
		curr_lp = prop_lp
	return beta, curr_lp

def calc_post(init_beta, label="", color='b', seed=100, reg=1e-42):
	n_iter = howlong
	n_iter = int(n_iter//8)

	rng = np.random.default_rng(seed)
	beta = np.copy(init_beta)
	new_metric = None
	covariance = None

	adaptive_epsilon = epsilon


	for j in range(3):
		accepted = 0
		if j == 2:
			n_iter *= 2
			#1 1 2
		curr_lp = log_posterior(beta)
		mean_beta = np.zeros(num_params)
		betas = np.zeros((n_iter, num_params))
		for i in range(n_iter):
			old_beta = beta
			beta, curr_lp = mh(beta, curr_lp, adaptive_epsilon, new_metric, covariance)
			mean_beta += beta
			betas[i] = beta

		mean_beta /= n_iter
		#sample_sd = np.sqrt(np.sum((betas - mean_beta)**2, axis=0)/n_iter)
		covariance = np.zeros((num_params, num_params))
		for beta in betas:
			covariance += np.outer(beta - mean_beta, beta - mean_beta)
		covariance /= n_iter
		covariance += reg * np.identity(num_params)
		# print(mean_beta)
		# print(covariance)
		new_metric = np.linalg.inv(covariance)


		if accepted < 0.7*n_iter:
			adaptive_epsilon *= 0.5
		else:
			adaptive_epsilon *= 2
	n_iter *= 2
	# 4
	accepted = 0
	mean_beta = np.zeros(num_params)
	betas = np.zeros((n_iter, num_params))
	for i in range(n_iter):
		old_beta = beta
		beta, curr_lp = mh(beta, curr_lp, adaptive_epsilon, new_metric, covariance)
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

ah_acpts = np.zeros(num_walks)
ah_means = np.zeros((num_params, num_walks))
ah_errs = np.zeros((num_params, num_walks))


for i in range(num_walks):
	rng = np.random.default_rng(i)
	# x = rng.normal(size=(n, num_params))
	# x[:,1] = 1
	# p = sigmoid(np.dot(x, true_beta))
	# y = p > rng.random(n)#rng.binomial(1, p)

	a, b, c = calc_post(init_beta)
	ah_acpts[i] = a
	ah_means[:, i] = b
	ah_errs[:, i] = c



propose = lambda beta, _, __, ___: rng.normal(loc=beta, scale = sqrt(0.5))
g2_acpts = np.zeros(num_walks)
g2_means = np.zeros((num_params, num_walks))
g2_errs = np.zeros((num_params, num_walks))


for i in range(num_walks):
	rng = np.random.default_rng(i)
	# x = rng.normal(size=(n, num_params))
	# x[:,1] = 1
	# p = sigmoid(np.dot(x, true_beta))
	# y = p > rng.random(n)#rng.binomial(1, p)

	a, b, c = calc_post(init_beta)
	g2_acpts[i] = a
	g2_means[:, i] = b
	g2_errs[:, i] = c


#import numpy as np
from math import sqrt, pi, prod, log, exp
#from matplotlib import pyplot as plt

n=100
num_bits=3
reg=1e-42

rng = np.random.default_rng(6857693485739845578)
sigmoid = lambda x: 1 / (1 + np.exp(-x))
sample = lambda input: input > rng.random(size=input.shape)
st_dev = lambda mean, samples: np.sqrt(np.sum((samples - mean)**2, axis=0)/len(samples))
def cov(mean, samples):
	covariance = np.zeros((num_params, num_params))
	beta = mean
	for beta in samples:
		tmp = beta - mean
		covariance += np.outer(tmp, tmp)
	covariance /= len(samples)
	covariance += reg * np.identity(len(beta))
	return covariance


num_params = len(true_beta)
# x = rng.normal(size=(n, num_params))
# x[:,0] = 1
# p = sigmoid(np.dot(x, true_beta))
# y = sample(p)

# prior_sd = 10
# prior_mean = (1<<(num_bits-1)) - 0.5
prior_mean=0
init_beta = rng.normal(loc=prior_mean, scale=prior_sd, size=num_params)

log_prior = lambda beta: -sum((beta - prior_mean)**2)/(2*prior_sd**2) - num_params*log(prior_sd*sqrt(2*pi))
grad_log_prior = lambda beta: (prior_mean - beta) / prior_sd**2
log_likelihood = lambda x_dot_beta: np.dot(y, x_dot_beta) - sum(np.log(1 + np.exp(x_dot_beta)))
log_posterior = lambda beta: log_likelihood(np.dot(x, beta)) + log_prior(beta)
grad_log_like = lambda beta: np.dot(y, x) - np.sum(x/(1+np.exp(-np.dot(x, beta))).reshape(-1,1), axis=0)
grad_log_post = lambda beta: grad_log_like(beta) + grad_log_prior(beta)

bias = np.zeros(num_params)
sample_betas = lambda bias, x_in, y_in: sample(sigmoid(bias + np.dot(y_in, x_in)/len(y_in)))
gen_betas = lambda bias, x_in, y_in: ( sample_betas(bias, x_in, y_in << i) << i for i in range(num_bits) )
def sample_betas_bits(bias, x_in, y_in):
	return np.sum( np.fromiter( gen_betas(bias, x_in, y_in) , f'({num_params},)u8', num_bits), axis=0, dtype=np.int16)

simple_propose = lambda beta: rng.normal(loc=beta, scale = sqrt(0.5))
uniform_propose = lambda _: rng.integers(1<<num_bits, size=num_params)
def single_dual_propose(_):
	i = rng.integers(n)
	return sample_betas_bits(bias, np.array([x[i]]), np.array([y[i]]))
global_dual_propose = lambda _: sample_betas_bits(bias, x, y)

def hmc_propose(beta, M, M_inv):
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

def mh(beta, curr_lp, propose=simple_propose):
	prop_beta = propose(beta)
	prop_lp = log_posterior(prop_beta)
	diff_lp = prop_lp - curr_lp
	if diff_lp > 0 or rng.random() < exp(diff_lp):
		beta = prop_beta
		curr_lp = prop_lp
	return beta, curr_lp

def loop(propose, n_iter, beta):
	total_beta = np.zeros(num_params)
	betas = np.zeros((n_iter, num_params))
	accepted = 0
	curr_lp = log_posterior(beta)
	for i in range(n_iter):
		old_beta = beta
		beta, curr_lp = mh(beta, curr_lp, propose)
		total_beta += beta
		betas[i] = beta
		if (beta != old_beta).any():
			accepted += 1
	return total_beta/n_iter, accepted/n_iter, betas

def calc_post(propose, init_beta, label="", color='b', seed=100):
	n_iter = int(howlong/8)
	#func_rng = np.random.default_rng(seed)
	curr_beta = np.copy(init_beta)
	new_metric = None
	covariance = None
	accepted = ()
	for j in range(3):
		if j > 1:
			n_iter *= 2
			#1 1 2 | 4
		partial_propose = lambda beta: propose(beta, new_metric, covariance)
		mean_beta, acc, betas = loop(partial_propose, n_iter, curr_beta)
		accepted += acc,
		covariance = cov(mean_beta, betas)
		new_metric = np.linalg.inv(covariance)
		curr_beta = mean_beta
	n_iter *= 2
	# 4
	partial_propose = lambda beta: propose(beta, new_metric, covariance)
	mean_beta, acc, betas = loop(partial_propose, n_iter, curr_beta)
	accepted += acc,
	# for i in range(num_params):
	# 	plt.figure(i)
	# 	plt.plot(betas[:,i], label=label)
	# 	plt.axhline(mean_beta[i], color=color)
	# 	plt.fill_between(range(-int(n_iter/20), n_iter+int(n_iter/20)), mean_beta[i]-sample_sd[i], mean_beta[i]+sample_sd[i], alpha=.2)
	return accepted, mean_beta, st_dev(mean_beta, betas)


from mod_q_sampler import GibbsSampler

v_qubits = num_params*num_bits
sampler = GibbsSampler(v_qubits, 1)
scaled_x = lambda x_sample: np.array([np.fromiter( ( x_sample * (2**i) for i in range(num_bits) ), dtype =f'({num_params},)f', count=num_bits).flatten()]).T

def reconstruct(full_state):
	hidden = full_state[0]
	visible = np.zeros(num_params, dtype=np.int16)
	for i in range(num_bits):
		visible += full_state[1+i*num_params:1+(i+1)*num_params] << i
	return hidden, visible

def convert_result(result):
	initial_state = 0
	for i in range(1, v_qubits+1):
		initial_state ^= result[-i] << (i-1)
	return initial_state

def rescale(mean_beta, width):
	min_beta = mean_beta - width
	Ls = 2*width/ (2**num_bits - 1)
	beta_unscaler = lambda b: Ls*b + min_beta
	new_mean_beta = ((mean_beta - min_beta)/Ls).astype(np.uint8)
	y_biases = np.dot(x, min_beta)
	new_x = np.dot(x, Ls)
	return beta_unscaler, new_mean_beta, y_biases, new_x
	#return new_x, mins, Ls


def q_loop(n_iter, lr=0.0, new_x=x, y_biases=np.zeros(n), beta_unscaler = lambda beta: beta):
	bias = np.zeros(num_params)
	total_betas = np.zeros(num_params)
	betas = np.zeros((n_iter, num_params))
	accepted = 0
	initial_state = rng.integers(1<<v_qubits)
	curr_lp = -100000
	curr_betas = None

	for j in range(n_iter):
		i = rng.integers(n)
		x_sample = scaled_x(new_x[i])
		sampler.convert(0.00000001+np.zeros((v_qubits,1)), np.array([y_biases[i]]), np.dot(x_sample, [y[i]]))
		initial_state |= y[i]<<v_qubits
		result = sampler.one_shot(1, initial_state, False)
		h, prop_betas = reconstruct(result)
		prop_lp = log_posterior(beta_unscaler(prop_betas))
		if sample(np.exp(prop_lp - curr_lp)):
			if curr_betas is not None:
				bias += lr*(prop_betas - curr_betas)
			curr_betas = prop_betas
			curr_lp = prop_lp
			accepted += 1
			initial_state = convert_result(result)
		total_betas += curr_betas
		betas[j] = curr_betas

	mean_beta = total_betas / n_iter

	return accepted / n_iter, mean_beta, st_dev(mean_beta, betas), betas


q_acpts = np.zeros(num_walks)
q_means = np.zeros((num_params, num_walks))
q_errs = np.zeros((num_params, num_walks))

for j in range(num_walks):
	betas = np.zeros((howlong, num_params))
	rng = np.random.default_rng(j)
	n_iter = howlong//4
	# x = rng.normal(size=(n, num_params))
	# x[:,1] = 1
	# p = sigmoid(np.dot(x, true_beta))
	# y = p > rng.random(n)#rng.binomial(1, p)
	beta_unscaler, new_mean_beta, y_biases, new_x = rescale(np.zeros(num_params), 3*np.ones(num_params))
	acc, scaled_mean, scaled_sd, scaled_betas = q_loop(n_iter, 0.0, new_x, y_biases, beta_unscaler)
	for i in range(n_iter):
		betas[i] = beta_unscaler(scaled_betas[i])
	mean_beta = beta_unscaler(scaled_mean)
	sd_beta = beta_unscaler(scaled_sd) - beta_unscaler(0)
	#print("biased q dual sample 1", acc, mean_beta, sd_beta)
	beta_unscaler, new_mean_beta, y_biases, new_x = rescale(mean_beta, 4*sd_beta)# + max(sd_beta))
	acc, scaled_mean, scaled_sd, scaled_betas = q_loop(n_iter, 0.0, new_x, y_biases, beta_unscaler)
	mean_beta = beta_unscaler(scaled_mean)
	sd_beta = beta_unscaler(scaled_sd) - beta_unscaler(0)
	for i in range(n_iter):
		betas[n_iter + i] = beta_unscaler(scaled_betas[i])
	n_iter *= 2
	beta_unscaler, new_mean_beta, y_biases, new_x = rescale(mean_beta, 3*sd_beta)
	acc, scaled_mean, scaled_sd, scaled_betas = q_loop(n_iter, 0.0, new_x, y_biases, beta_unscaler)
	mean_beta = beta_unscaler(scaled_mean)
	sd_beta = beta_unscaler(scaled_sd) - beta_unscaler(0)
	for i in range(n_iter):
		betas[n_iter + i] = beta_unscaler(scaled_betas[i])


	plt.figure(5)
	plt.clf()

	fig, axes = plt.subplots(num_params, sharex=True)
	fig.supxlabel("iterations")
	fig.supylabel("parameter value")
	fig.suptitle(f"{acc} acceptance rate")

	for i in range(num_params):
		ax = axes[i]
		ax.plot(betas[:,i])#, label=label)
		ax.axhline(mean_beta[i])#, color=color)
		ax.axhline(true_beta[i], color='m')
		ax.fill_between(range(-int(howlong/20), howlong+int(howlong/20)), mean_beta[i]-sd_beta[i], mean_beta[i]+sd_beta[i], alpha=.2)
	
	#print("biased q dual sample 2", acc, mean_beta, sd_beta)
	a, b, c = acc, mean_beta, sd_beta

	q_acpts[j] = a
	q_means[:, j] = b
	q_errs[:, j] = c


plt.figure(6)
fig, axes = plt.subplots(1,4, sharey=True)
fig.supylabel("acceptance rate")
fig.supxlabel("parameter value")
axes[0].set_title("bias, x=1")

for i in range(num_params):
	#plt.figure(i)
	axs = axes[i]
	#plt.errorbar(ah_means[i], ah_acpts, xerr=ah_errs[i], label="Adaptive HMC", marker='.', linestyle='', alpha=0.4)
	axs.errorbar(h_means[i], h_acpts, xerr=h_errs[i], label=f"Hamiltonian MC", marker='.', linestyle='', alpha=0.4)
	#plt.errorbar(h_means[i], h_acpts, xerr=h_errs[i], label=f"Hamiltonian MC, with {bunk_h} divergent", marker='.', linestyle='', alpha=0.4)
	axs.errorbar(g2_means[i], g2_acpts, xerr=g2_errs[i], label="Gaussian MH", marker='.', linestyle='', color='purple', alpha=0.4)
	#plt.errorbar(g_means[i], g_acpts, xerr=g_errs[i], marker='.', linestyle='', color='purple', alpha=0.4)
	axs.errorbar(q_means[i], q_acpts, xerr=q_errs[i], label="Quantum", marker='.', linestyle='', alpha=0.4)

	axs.axvline(true_beta[i], color='m', label="true value")
	# axs.set_xlabel("parameter")
	#axs.ylabel("acceptance rate")
axs.legend()


print(abs(sum(ah_means[:,i] for i in range(num_walks))/num_walks  - true_beta))
print(abs(sum(h_means[:,i] for i in range(num_walks))/num_walks  - true_beta))
print(abs(sum(g_means[:,i]+g2_means[:,i] for i in range(num_walks))/(2*num_walks) - true_beta))
print(bunk_h/num_walks)

#[2.36308818 4.17396113 1.73040232 2.36392416]

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

