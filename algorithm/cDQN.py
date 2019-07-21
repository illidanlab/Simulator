

import numpy as np
import tensorflow as tf
import random, os
from alg_utility import *

# this is essentially deep expected SARSA.
class Estimator:
    """ build value network
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

            self.loss_gradients = tf.gradients(self.loss, tf.trainable_variables(scope=scope))
                                           # tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))
        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.scalar("value_output", tf.reduce_mean(self.value_output)),
            tf.summary.scalar("gradient_norm_policy", tf.reduce_sum([tf.norm(item) for item in self.loss_gradients]))
        ])

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
        self.y_pl = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="y")

        self.loss_lr = tf.placeholder(tf.float32, None, "learning_rate")

        # 3 layers feed forward network.
        l1 = fc(X, "l1", 128, act=tf.nn.relu)
        l2 = fc(l1, "l2", 64, act=tf.nn.relu)
        l3 = fc(l2, "l3", 32, act=tf.nn.relu)
        # l1 = tf.layers.dense(X, 1024, tf.nn.sigmoid, trainable=trainable)
        # l2 = tf.layers.dense(l1, 512, tf.nn.sigmoid, trainable=trainable)
        # l3 = tf.layers.dense(l2, 32, tf.nn.sigmoid, trainable=trainable)
        self.value_output = fc(l3, "value_output", 1, act=tf.nn.relu)

        # self.losses = tf.square(self.y_pl - self.value_output)
        self.loss = tf.reduce_mean(tf.squared_difference(self.y_pl, self.value_output))

        self.train_op = tf.train.AdamOptimizer(self.loss_lr).minimize(self.loss)


    def predict(self, s):
        value_output = self.sess.run(self.value_output, {self.state: s})

        return value_output

    def action(self, s, context, epsilon):
        """ Compute current action for all grids give states

        :param s: 504 x stat_dim,
        :return:
        """

        # value of each grid at next time step, dispatched according to this value.
        value_output = self.sess.run(self.value_output, {self.state: s}).flatten()
        action_tuple = []
        valid_prob = []

        grid_ids = np.argmax(s[:, -self.n_valid_grid:], axis=1)

        for idx, grid_valid_idx in enumerate(grid_ids):
            valid_qvalues = value_output[self.neighbors_list[grid_valid_idx]]
            temp_qvalue = np.zeros(self.action_dim)

            if np.sum(valid_qvalues) == 0:
                # all value equals to 0. this could explores conflicts action.
                temp_qvalue[self.valid_action_mask[grid_valid_idx] > 0] = 1. / np.sum(
                    self.valid_action_mask[grid_valid_idx])
                action_prob = temp_qvalue
                valid_prob.append(action_prob)
            else:

                temp_qvalue[self.valid_action_mask[grid_valid_idx] > 0] = valid_qvalues
                temp_qvalue[temp_qvalue < temp_qvalue[-1]] = 0

                best_action = np.argmax(temp_qvalue)
                num_valid_action = np.count_nonzero(temp_qvalue)
                action_prob = np.zeros_like(temp_qvalue)
                action_prob[temp_qvalue > 0] = epsilon / float(num_valid_action)
                action_prob[best_action] += 1 - epsilon
                valid_prob.append(action_prob)

            if int(context[idx]) == 0:
                continue
            curr_action_indices = np.random.multinomial(int(context[idx]),
                                                        action_prob)

            start_node_id = self.env.target_grids[grid_valid_idx]
            for curr_action_idx, num_driver in enumerate(curr_action_indices):
                if num_driver > 0:
                    end_node_id = int(self.valid_neighbor_node_id[grid_valid_idx, curr_action_idx])
                    if end_node_id != start_node_id:
                        action_tuple.append((start_node_id, end_node_id, num_driver))

        return action_tuple, np.stack(valid_prob)

    def compute_targets(self, valid_prob, next_state, node_reward, gamma):
        targets = []
        node_reward = node_reward.flatten()
        qvalue_next = self.sess.run(self.value_output, {self.state: next_state}).flatten()  # qvalue of next state

        for idx in np.arange(self.n_valid_grid):
            grid_prob = valid_prob[idx][self.valid_action_mask[idx]>0]
            neighbor_grid_ids = self.neighbors_list[idx]
            best_grid = np.argmax(grid_prob)
            curr_grid_target = node_reward[neighbor_grid_ids][best_grid] + gamma * qvalue_next[neighbor_grid_ids][best_grid]
            targets.append(curr_grid_target)

        return np.array(targets).reshape([-1, 1])

    def initialization(self, s, y, learning_rate):
        sess = self.sess
        feed_dict = {self.state: s, self.y_pl: y, self.loss_lr: learning_rate}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def update(self, s, y, learning_rate, global_step):
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
        feed_dict = {self.state: s, self.y_pl: y, self.loss_lr: learning_rate}
        summaries, _, loss = sess.run([self.summaries, self.train_op, self.loss], feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.flush()
        return loss

    def _build_cnn_model(self):

        # states of grid id and time
        self.state_spacetime = Xst = tf.placeholder(shape=[None, self.n_valid_grid + self.T], dtype=tf.uint8, name="Xst")

        # states of distribution
        self.state = X = tf.placeholder(shape=[None, self.M, self.N,  4], dtype=tf.uint8, name="X")

        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

        batch_size = tf.shape(self.state)[0]

        conv1 = tf.contrib.layers.conv2d(
            X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, self.action_dim)


class stateProcessor:
    """
        Process a raw global state into the states of grids.
    """

    def __init__(self,
                 target_id_states,
                 target_grids,
                 n_valid_grids):
        self.target_id_states = target_id_states  # valid grid index for driver and order distribution.
        self.target_grids = target_grids   # valid grid id [22, 24, ...]
        self.n_valid_grids = n_valid_grids
        self.T = 144
        self.action_dim = 7
        self.extend_state = True

    def utility_conver_states(self, curr_state):
        curr_s = np.array(curr_state).flatten()
        curr_s_new = [curr_s[idx] for idx in self.target_id_states]
        return np.array(curr_s_new)

    def utility_normalize_states(self, curr_s):
        max_driver_num = np.max(curr_s[:self.n_valid_grids])
        max_order_num = np.max(curr_s[self.n_valid_grids:])

        curr_s_new = np.zeros_like(curr_s)
        curr_s_new[:self.n_valid_grids] = curr_s[:self.n_valid_grids] / max_driver_num
        curr_s_new[self.n_valid_grids:] = curr_s[self.n_valid_grids:] / max_order_num
        return curr_s_new

    def utility_conver_reward(self, reward_node):
        reward_node_new = [reward_node[idx] for idx in self.target_grids]
        return np.array(reward_node_new)

    def reward_wrapper(self, info, curr_s):
        info_reward = info[0]
        valid_nodes_reward = self.utility_conver_reward(info_reward[0])
        devide = curr_s[:self.n_valid_grids]
        devide[devide == 0] = 1
        valid_nodes_reward = valid_nodes_reward/devide
        return valid_nodes_reward

    def compute_context(self, info):
        # 计算context
        context = info.flatten()
        context = [context[idx] for idx in self.target_grids]
        return context

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

    def to_grid_rewards(self, node_reward):
        return np.array(node_reward).reshape([-1, 1])

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




