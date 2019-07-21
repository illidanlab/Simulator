import tensorflow as tf
import numpy as np
# import cvxpy as cvx
from simulator.utilities import *

""" Some of Following codes are modified from https://github.com/openai/baselines
"""
def tfsum(x, axis=None, keepdims=False):
    axis = None if axis is None else [axis]
    return tf.reduce_sum(x, axis=axis, keepdims=keepdims)

class Pd(object):
    """
    A particular probability distribution
    """

    def mode(self):
        raise NotImplementedError

    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def logp(self, x):
        return - self.neglogp(x)


class DiagGaussianPd(Pd):
    def __init__(self, mu, logstd):
        self.mean = mu
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def mode(self):
        return self.mean

    def neglogp(self, x):
        # axis = -1, sum over last dimension, first dimension is batch size
        return 0.5 * tfsum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tfsum(self.logstd, axis=-1)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))



def normalize_reward(discounted_epr):
    reward_mean = np.mean(discounted_epr)
    reward_std  = np.std(discounted_epr)
    discounted_epr = (discounted_epr - reward_mean)/reward_std
    return discounted_epr

# def projection(Y, n, num_idle_driver):
#     assert np.sum(Y) > num_idle_driver
#     X = cvx.Variable(n)
#     objective = cvx.Minimize(cvx.sum_squares(X - Y))
#     constraints = [0 <= X,
#                    num_idle_driver == cvx.sum_entries(X)]
#     prob = cvx.Problem(objective, constraints)
#
#     # The optimal objective is returned by prob.solve().
#     result = prob.solve()
#     return X.value


def continuous_quadratic_knapsack(b, u, r):
    """
    OBJECTIVE
     min 1/2*||x||_2^2
     s.t. b'*x = r, 0<= x <= u,  b > 0
    
    Related paper
     [1] KC. Kiwiel. On linear-time algorithms for the continuous 
         quadratic knapsack problem, Journal of Optimization Theory 
         and Applications, 2007
    Coding Reference: 
    https://github.com/jiayuzhou/MALSAR/blob/master/MALSAR/functions/CMTL/bsa_ihb.m
    """
    n = len(b)
    break_flag = 0 
    t_l = np.zeros(n)
    t_u = -u    # ( 0 - u)/1
    t_L = -float('Inf')
    t_U = float('Inf')
    g_tL = 0
    g_tU = 0
    T = np.concatenate((t_l, t_u), axis=0)
    n_iter = 0
    while len(T) !=0:
        n_iter += 1
        g_t = 0
        t_hat = np.median(T)

        U_inds = np.where(t_hat < t_u)
        M      = np.where((t_u <= t_hat) & (t_hat <= t_l))

        if len(U_inds[0]) != 0: 
            g_t = g_t  + np.dot(b[U_inds], u[U_inds])

        if len(M[0]) != 0:
            g_t = g_t - np.dot(b[M], t_hat*b[M])  # a = 0   np.sum(b(M).*(a(M) - t_hat*b(M)))
        if g_t > r:
            t_L = t_hat
            T = T[np.where(T > t_hat)]
            g_tL = g_t 
        elif g_t < r:
            t_U = t_hat
            T = T[np.where(T < t_hat)]
            g_tU = g_t
        else:
            t_star = t_hat
            break_flag = 1
            break
            
    if break_flag == 0:
         t_star = t_L - (g_tL -r)*(t_U - t_L)/(g_tU - g_tL)
    x_star = np.minimum(np.maximum(0, -t_star*b), u)
    return x_star


def projection_fast(u, n, num_idle_driver):
    b = np.ones((n))
    r = np.sum(u) - num_idle_driver
    x_star = continuous_quadratic_knapsack(b, u, r)
    return u - x_star


def categorical_sample_split(logits, d=6):
    """
    :param logits: sampling according to the probability exp(logits)
    :param d: first dimension of logits. 6 in our case.
    :return:
    """

    value = [tf.multinomial(logits[i] - tf.reduce_max(logits[i], [1], keep_dims=True), 1)
             for i in np.arange(d)
             ]
    return value

def fc(x, scope, nh, act=tf.nn.relu, init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        # w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        w = tf.get_variable("w", [nin, nh], initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=0))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w)+b
        h = act(z)
        return h


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        np.random.seed(1)
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


##############################################################
### utility function for tune DQN
##############################################################

def get_target_ids_local(mapped_matrix_int_local):
    row_inds, col_inds = np.where(mapped_matrix_int_local >= 0)

    M_local, N_local = mapped_matrix_int_local.shape
    target_ids_local  = []  # start from 0.
    for x, y in zip(row_inds, col_inds):
        node_id = ids_2dto1d(x, y, M_local, N_local)
        target_ids_local.append(node_id)
    return target_ids_local

def collision_action(action_tuple):
    count = 0
    action_set = set(())
    for item in action_tuple:
        if item[1] == -1:
            continue
        grid_id_key = str(item[0]) + "-" + str(item[1])
        action_set.add(grid_id_key)
        conflict_id_key = str(item[1]) + "-" + str(item[0])
        if conflict_id_key in action_set:
            count += 1
    return count

def construct_grid_nodeid_mapping(target_ids_local, grid_ids_local):
    node_mapping = {}
    grid_mapping = {} #
    for nodeid, gridid in zip(target_ids_local, grid_ids_local):
        node_mapping[gridid] = nodeid
        grid_mapping[nodeid] = gridid
    return node_mapping, grid_mapping

def utility_conver_states(curr_s, target_id_states):
    curr_s_new = [curr_s[idx] for idx in target_id_states]
    return np.array(curr_s_new)

def utility_conver_reward(reward_node, target_id_states):
    reward_node_new = [reward_node[idx] for idx in target_id_states]
    return np.array(reward_node_new)

##############################################################


def compute_sum_qtable(temp_qtable):
    temp_q = 0
    for item in temp_qtable:
        for jj in item:
            temp_q += np.sum(jj)

    return temp_q