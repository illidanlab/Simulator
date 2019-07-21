
import pickle, sys
sys.path.append("../")

# from simulator.utilities import *
from algorithm.alg_utility import *
from simulator.envs import *

################## Load data ###################################
dir_prefix = "/mnt/research/linkaixi/AllData/dispatch/"
current_time = time.strftime("%Y%m%d_%H-%M")
log_dir = dir_prefix + "dispatch_simulator/experiments/{}/".format(current_time)
mkdir_p(log_dir)
print "log dir is {}".format(log_dir)


data_dir = dir_prefix + "dispatch_realdata/data_for_simulator/"
order_time_dist = []
order_price_dist = []
mapped_matrix_int = pickle.load(open(data_dir+"mapped_matrix_int.pkl", 'rb'))
order_num_dist = pickle.load(open(data_dir+"order_num_dist", 'rb'))
idle_driver_dist_time = pickle.load(open(data_dir+"idle_driver_dist_time", 'rb'))
idle_driver_location_mat = pickle.load(open(data_dir+"idle_driver_location_mat", 'rb'))
target_ids = pickle.load(open(data_dir+"target_grid_id.pkl", 'rb'))
onoff_driver_location_mat = pickle.load(open(data_dir + "onoff_driver_location_mat", 'rb'))
order_filename = dir_prefix + "dispatch_realdata/orders/all_orders_target"
order_real = pickle.load(open(order_filename, 'rb'))
M, N = mapped_matrix_int.shape
print "finish load data"


################## Initialize env ###################################
n_side = 6
GAMMA = 0.9
l_max = 9

env = CityReal(mapped_matrix_int, order_num_dist,
               idle_driver_dist_time, idle_driver_location_mat,
               order_time_dist, order_price_dist,
               l_max, M, N, n_side, 1/28.0, order_real, onoff_driver_location_mat)


log_dir = dir_prefix + "dispatch_simulator/experiments/{}/".format(current_time)


temp = np.array(env.target_grids) + env.M * env.N
target_id_states = env.target_grids + temp.tolist()


curr_s = np.array(env.reset_clean()).flatten()  # [0] driver dist; [1] order dist
curr_s = utility_conver_states(curr_s, target_id_states)
print "******************* Finish generating one day order **********************"



print "******************* Starting training Deep SARSA **********************"
from algorithm.cDQN import *

MAX_ITER = 50
is_plot_figure = False
city_time_start = 0
EP_LEN = 144
global_step = 0
city_time_end = city_time_start + EP_LEN
EPSILON = 0.8
gamma = 0.9
learning_rate = 1e-3

prev_epsiode_reward = 0
curr_num_actions = []
all_rewards = []
order_response_rate_episode = []
value_table_sum = []
episode_rewards = []
episode_conflicts_drivers = []
record_all_order_response_rate = []

T = 144
action_dim = 7
state_dim = env.n_valid_grids * 3 + T


# tf.reset_default_graph()
sess = tf.Session()
tf.set_random_seed(1)

q_estimator = Estimator(sess, action_dim,
                        state_dim,
                        env,
                        scope="q_estimator",
                        summaries_dir=log_dir)

target_estimator = Estimator(sess, action_dim, state_dim, env, scope="target_q")
sess.run(tf.global_variables_initializer())
estimator_copy = ModelParametersCopier(q_estimator, target_estimator)
replay = ReplayMemory(memory_size=1e+6, batch_size=int(3e+3))
stateprocessor = stateProcessor(target_id_states, env.target_grids, env.n_valid_grids)

RATIO = 1
saver = tf.train.Saver()


print "Start training contextual deep Q learning. "
save_random_seed = []
N_ITER_RUNS = 25
temp_value = 10
EPSILON_start = 0.5
EPSILON_end = 0.1
epsilon_decay_steps=15
epsilons = np.linspace(EPSILON_start, EPSILON_end, epsilon_decay_steps)
episode_dispatched_drivers = []
for n_iter in np.arange(N_ITER_RUNS):
    RANDOM_SEED = n_iter + MAX_ITER - temp_value
    env.reset_randomseed(RANDOM_SEED)
    save_random_seed.append(RANDOM_SEED)
    batch_s, batch_a, batch_r = [], [], []
    batch_reward_gmv = []
    epsiode_reward = 0
    num_dispatched_drivers = 0

    # reset env
    # if n_iter % 1 == 0:
    #     is_regenerate_order = 1
    # else:
    #     is_regenerate_order = 0
    is_regenerate_order = 1
    curr_state = env.reset_clean(generate_order=is_regenerate_order, ratio=RATIO, city_time=city_time_start)
    info = env.step_pre_order_assigin(curr_state)
    context = stateprocessor.compute_context(info)
    curr_s = stateprocessor.utility_conver_states(curr_state)
    normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
    s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0

    # record rewards to update the value table
    episodes_immediate_rewards = []
    order_response_rates = []
    # epsilon = EPSILON * (1 - np.max([n_iter, temp_value+5]) / N_ITER_RUNS)  #testing staget 不改变epsilon
    epsilon = epsilons[n_iter] if n_iter < 15 else EPSILON_end
    num_conflicts_drivers = []
    curr_num_actions = []
    for ii in np.arange(EP_LEN + 1):
        # INPUT: state,  OUTPUT: action
        action_tuple, valid_action_prob_mat = q_estimator.action(s_grid, context, epsilon)

        # a0

        # ONE STEP: r0
        next_state, r, info = env.step(action_tuple, 2)

        # r0
        immediate_reward = stateprocessor.reward_wrapper(info, curr_s)

        # Save transition to replay memory
        if ii != 0:
            # r1, c0
            r_grid = stateprocessor.to_grid_rewards(immediate_reward)
            # s0, a0, r1
            targets_batch = target_estimator.compute_targets(action_mat_prev, s_grid, r_grid, gamma)
            replay.add(state_mat_prev, action_mat_prev, targets_batch, s_grid)

        state_mat_prev = s_grid
        action_mat_prev = valid_action_prob_mat

        # s1
        curr_s = stateprocessor.utility_conver_states(next_state)
        normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
        s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0

        # c1
        context = stateprocessor.compute_context(info[1])

        # training method 2
        # Sample a minibatch from the replay memory and update q network
        # if replay.curr_lens != 0:
        #     for _ in np.arange(20):
        #         batch_s, _, batch_r, _ = replay.sample()
        #         iloss = q_estimator.update(batch_s, batch_r, 1e-3, global_step)
        #         global_step += 1
        # print "******** city time {} *********".format(env.city_time)

        # Perform gradient descent update
        # book keeping
        global_step += 1
        all_rewards.append(r)
        batch_reward_gmv.append(r)
        order_response_rates.append(env.order_response_rate)
        curr_num_action = np.sum([aa[2] for aa in action_tuple]) if len(action_tuple) != 0 else 0
        curr_num_actions.append(curr_num_action)
        num_conflicts_drivers.append(collision_action(action_tuple))

    # training method 1
    for _ in np.arange(4000):
        batch_s, _, batch_r, _ = replay.sample()
        iloss = q_estimator.update(batch_s, batch_r, 1e-3, global_step)
        global_step += 1

        # update target Q network
    #     if (global_step + 1) % 70 == 0:
    estimator_copy.make(sess)

    episode_reward = np.sum(batch_reward_gmv[1:])
    episode_rewards.append(episode_reward)
    n_iter_order_response_rate = np.mean(order_response_rates[1:])
    order_response_rate_episode.append(n_iter_order_response_rate)
    record_all_order_response_rate.append(order_response_rates)
    episode_conflicts_drivers.append(np.sum(num_conflicts_drivers[:-1]))
    episode_dispatched_drivers.append(np.sum(curr_num_actions[:-1]))


    print "******** iteration {} ********* reward {}, order_response_rate {} number drivers {}, conflicts {}, epsilon {}".format(n_iter, episode_reward,
                                                                                                        n_iter_order_response_rate,
                                                                                                         episode_dispatched_drivers[-1],
                                                                                                        episode_conflicts_drivers[-1],
                                                                                                             epsilon)


    pickle.dump([episode_rewards, order_response_rate_episode, save_random_seed, episode_conflicts_drivers, episode_dispatched_drivers], open(log_dir + "results.pkl", "w"))

    saver.save(sess, log_dir+"model.ckpt")


