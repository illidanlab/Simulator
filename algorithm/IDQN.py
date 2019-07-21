
import numpy as np
import tensorflow as tf
import random, os
from alg_utility import *


class Estimator:
    """ build Deep Q network
    """
    def __init__(self,
                 sess,
                 action_dim,
                 state_dim,
                 env,
                 scope="estimator",
                 summaries_dir=None):
        self.sess = sess
        self.n_valid_grid = env.n_valid_grids
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.M = env.M
        self.N = env.N
        self.scope = scope
        self.T = 144
        self.env = env

        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

        self.neighbors_list = []
        for idx, node_id in enumerate(env.target_grids):
            neighbor_indices = env.nodes[node_id].layers_neighbors_id[0]  # index in env.nodes
            neighbor_ids = [env.target_grids.index(env.nodes[item].get_node_index()) for item in neighbor_indices]
            neighbor_ids.append(idx)
            # index in env.target_grids == index in state
            self.neighbors_list.append(neighbor_ids)

        # compute valid action mask.
        self.valid_action_mask = np.ones((self.n_valid_grid, self.action_dim))
        self.valid_neighbor_node_id = np.zeros((self.n_valid_grid, self.action_dim))  # id in env.nodes
        self.valid_neighbor_grid_id = np.zeros((self.n_valid_grid, self.action_dim))  # id in env.target_grids
        for grid_idx, grid_id in enumerate(env.target_grids):
            for neighbor_idx, neighbor in enumerate(self.env.nodes[grid_id].neighbors):
                if neighbor is None:
                    self.valid_action_mask[grid_idx, neighbor_idx] = 0
                else:
                    node_index = neighbor.get_node_index()  # node_index in env.nodes
                    self.valid_neighbor_node_id[grid_idx, neighbor_idx] = node_index
                    self.valid_neighbor_grid_id[grid_idx, neighbor_idx] = env.target_grids.index(node_index)

            self.valid_neighbor_node_id[grid_idx, -1] = grid_id
            self.valid_neighbor_grid_id[grid_idx, -1] = grid_idx



    def _build_model(self):
        trainable = True
        self.state = X = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="X")

        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

        # action chosen
        self.ACTION = tf.placeholder(tf.float32, [None, self.action_dim], 'action_chosen')

        self.loss_lr = tf.placeholder(tf.float32, None, "learning_rate")

        # 3 layers feed forward network.
        l1 = tf.layers.dense(X, 128, tf.nn.elu, trainable=trainable)
        l2 = tf.layers.dense(l1, 64, tf.nn.elu, trainable=trainable)
        l3 = tf.layers.dense(l2, 32, tf.nn.elu, trainable=trainable)
        self.qvalue = tf.layers.dense(l3, self.action_dim, tf.nn.elu, trainable=trainable)


        # get the Q(s,a) for chosen action
        self.action_predictions = tf.reduce_sum(self.qvalue * self.ACTION, axis=1)

        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        self.train_op = tf.train.AdamOptimizer(self.loss_lr).minimize(self.loss)

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.qvalue),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.qvalue))
        ])

    def predict(self, s):
        qvalues = self.sess.run(self.qvalue, {self.state: s})
        max_qvalue = np.max(qvalues, axis=1)
        return max_qvalue

    def action(self, s, context, epsilon):
        """ Compute Q(s, a) for all actions give states
        :return:
        """
        # A = np.ones(self.action_dim, dtype=float) * epsilon / self.action_dim
        qvalues = self.sess.run(self.qvalue, {self.state: s})
        action_idx = []            # go to which node, the index in nodes
        action_idx_valid = []            #the index in env.target_grids
        action_neighbor_idx = []
        action_tuple_mat = np.zeros((len(self.env.nodes), len(self.env.nodes)))
        action_tuple = []
        action_starting_gridids = []

        grid_ids = np.argmax(s[:, -self.n_valid_grid:], axis=1)  # starting grid of each sample
        valid_probs = []

        for idx, grid_valid_idx in enumerate(grid_ids):
            curr_qvalue = qvalues[idx]
            temp_qvalue = self.valid_action_mask[grid_valid_idx] * curr_qvalue

            if np.sum(temp_qvalue) == 0:  # encourage exploration
                temp_qvalue[self.valid_action_mask[grid_valid_idx]>0] = 1. / np.sum(self.valid_action_mask[grid_valid_idx])
                action_prob = temp_qvalue / np.sum(temp_qvalue)
                curr_action_indices = np.random.multinomial(int(context[idx]), action_prob)
            else:
                best_action = np.argmax(temp_qvalue)
                action_prob = np.zeros(self.action_dim)
                num_valid_action = np.count_nonzero(temp_qvalue)
                action_prob[temp_qvalue > 0] = epsilon / float(num_valid_action)
                action_prob[best_action] += 1 - epsilon
                curr_action_indices = np.random.multinomial(int(context[idx]), action_prob)

            valid_probs.append(action_prob)
            start_node_id = self.env.target_grids[grid_valid_idx]
            num_distinct_action = 0
            for curr_action_idx, num_driver in enumerate(curr_action_indices):
                if num_driver > 0:
                    end_node_id = int(self.valid_neighbor_node_id[grid_valid_idx, curr_action_idx])

                    action_idx.append(end_node_id)
                    action_idx_valid.append(int(self.valid_neighbor_grid_id[grid_valid_idx, curr_action_idx]))
                    action_neighbor_idx.append(curr_action_idx)
                    action_tuple_mat[start_node_id, end_node_id] = num_driver
                    num_distinct_action += 1
            action_starting_gridids.append(num_distinct_action)

        action_indices = np.where(action_tuple_mat > 0)
        for xx, yy in zip(action_indices[0], action_indices[1]):
            if xx != yy:
                action_tuple.append((xx, yy, int(action_tuple_mat[xx, yy])))

        return qvalues, action_idx, action_idx_valid, action_neighbor_idx, action_tuple, action_starting_gridids

    def update(self, s, a, y, learning_rate, global_step):
        """
        Updates the estimator towards the given targets.

        Args:
          s: State input of shape [batch_size, state_dim]
          a: Chosen actions of shape [batch_size, action_dim], 0, 1 mask
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        sess = self.sess
        feed_dict = {self.state: s, self.y_pl: y, self.ACTION: a, self.loss_lr: learning_rate}
        summaries, _, loss = sess.run([self.summaries, self.train_op, self.loss], feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss



class stateProcessor:
    """
        Process a raw global state into the states of grids.
    """

    def __init__(self,
                 target_id_states,
                 target_grids,
                 n_valid_grids):
        self.target_id_states = target_id_states  # valid grid index for driver and order distribution.
        self.target_grids = target_grids   # valid grid id [22, 24, ...]  504
        self.n_valid_grids = n_valid_grids
        self.T = 144
        self.action_dim = 7
        self.extend_state = True

    def utility_conver_states(self, curr_state):
        curr_s = np.array(curr_state).flatten()
        curr_s_new = [curr_s[idx] for idx in self.target_id_states]
        return np.array(curr_s_new)

    def utility_conver_reward(self, reward_node):
        reward_node_new = [reward_node[idx] for idx in self.target_grids]
        return np.array(reward_node_new)

    def reward_wrapper(self, info, curr_s):
        """
        :param info: [node_reward(including neighbors), neighbor_reward]
        :param curr_s:
        :return:
        """

        info_reward = info[0]
        valid_nodes_reward = self.utility_conver_reward(info_reward[0])
        devide = curr_s[:self.n_valid_grids]
        devide[devide == 0] = 1
        valid_nodes_reward = valid_nodes_reward/devide
        return valid_nodes_reward

    def compute_context(self, info):
        # compute context
        context = info.flatten()
        context = [context[idx] for idx in self.target_grids]
        return context

    def utility_normalize_states(self, curr_s):
        max_driver_num = np.max(curr_s[:self.n_valid_grids])
        max_order_num = np.max(curr_s[self.n_valid_grids:])

        curr_s_new = np.zeros_like(curr_s)
        curr_s_new[:self.n_valid_grids] = curr_s[:self.n_valid_grids] / max_driver_num
        curr_s_new[self.n_valid_grids:] = curr_s[self.n_valid_grids:] / max_order_num
        return curr_s_new


    def to_grid_states(self, curr_s, curr_city_time):
        T = self.T

        # curr_s = self.utility_conver_states(curr_state)
        time_one_hot = np.zeros((T))
        time_one_hot[curr_city_time % T] = 1
        onehot_grid_id = np.eye(self.n_valid_grids)

        s_grid = np.zeros((self.n_valid_grids, self.n_valid_grids * 3 + T))
        s_grid[:, :self.n_valid_grids * 2] = np.stack([curr_s] * self.n_valid_grids)
        s_grid[:, self.n_valid_grids * 2:self.n_valid_grids * 2 + T] = np.stack([time_one_hot] * self.n_valid_grids)
        s_grid[:, -self.n_valid_grids:] = onehot_grid_id
        return np.array(s_grid)

    def to_grid_rewards(self, action_idx_valid, node_reward):
        r_grid = []
        for end_grid_id in action_idx_valid:
            r_grid.append(node_reward[end_grid_id])

        return np.array(r_grid)


    def to_grid_next_states(self, s_grid, next_state, action_index, curr_city_time):
        """
        :param s_grid:  batch_size x state_dimension
        :param action_index: batch_size, end_valid_grid_id, next grid id.
        :return:
        """
        T = self.T
        next_s = self.utility_normalize_states(self.utility_conver_states(next_state))

        time_one_hot = np.zeros((T))
        time_one_hot[curr_city_time % T] = 1

        s_grid_next = np.zeros(s_grid.shape)
        s_grid_next[:, :self.n_valid_grids*2] = next_s
        s_grid_next[:, self.n_valid_grids*2:self.n_valid_grids*2+T] = time_one_hot

        action_index = np.array(action_index) + self.n_valid_grids*2 + T
        s_grid_next[np.arange(s_grid_next.shape[0]), action_index] = 1

        return s_grid_next

    def to_grid_state_for_training(self, s_grid, action_starting_gridids):
        s_grid_new = []
        for idx, num_extend in enumerate(action_starting_gridids):
            temp_s = s_grid[idx]
            s_grid_new += [temp_s] * num_extend
        return np.array(s_grid_new)

    def to_action_mat(self, action_neighbor_idx):
        action_mat = np.zeros((len(action_neighbor_idx), self.action_dim))
        action_mat[np.arange(action_mat.shape[0]), action_neighbor_idx] = 1
        return action_mat



class ReplayMemory:
    """ collect the experience and sample a batch for training networks.
        without time ordering
    """
    def __init__(self, memory_size, batch_size):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0

    def add(self, s, a, r, next_s):
        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.next_states = next_s
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s),axis=0)
            self.next_states = np.concatenate((self.next_states, next_s), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            new_sample_lens = s.shape[0]
            index = random.randint(0, self.curr_lens - new_sample_lens)

            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.next_states[index:(index + new_sample_lens)] = next_s

    def sample(self):

        if self.curr_lens <= self.batch_size:
            return [self.states, self.actions, self.rewards, self.next_states]
        indices = random.sample(range(0, self.curr_lens), self.batch_size)
        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices]
        batch_next_s = self.next_states[indices]
        return [batch_s, batch_a, batch_r, batch_next_s]

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.curr_lens = 0


class ModelParametersCopier():
    """
    Copy model parameters of one estimator to another.
    """

    def __init__(self, estimator1, estimator2):
        """
        Defines copy-work operation graph.
        Args:
          estimator1: Estimator to copy the paramters from
          estimator2: Estimator to copy the parameters to
        """
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)

    def make(self, sess):
        """
        Makes copy.
        Args:
            sess: Tensorflow session instance
        """
        sess.run(self.update_ops)




