
import pickle, sys
sys.path.append("../")

from simulator.envs import *


################## Load data ###################################
dir_prefix = "/mnt/research/linkaixi/AllData/dispatch/"
current_time = time.strftime("%Y%m%d_%H-%M")
log_dir = dir_prefix + "dispatch_simulator/experiments/{}/".format(current_time)
mkdir_p(log_dir)
print "log dir is {}".format(log_dir)

data_dir = dir_prefix + "dispatch_realdata/data_for_simulator2017-07-24_2017-08-20/"
order_time_dist = []
order_price_dist = []
mapped_matrix_int = pickle.load(open(data_dir+"mapped_matrix_int.pkl", 'rb'))
order_num_dist = pickle.load(open(data_dir+"order_num_dist", 'rb'))
idle_driver_dist_time = pickle.load(open(data_dir+"idle_driver_dist_time", 'rb'))
idle_driver_location_mat = pickle.load(open(data_dir+"idle_driver_location_mat", 'rb'))
target_ids = pickle.load(open(data_dir+"target_grid_id.pkl", 'rb'))
onoff_driver_location_mat = pickle.load(open(data_dir + "onoff_driver_location_mat", 'rb'))
order_filename = dir_prefix + "dispatch_realdata/order_new_2017-07-24_2017-08-20/all_orders_target"
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
mkdir_p(log_dir)

temp = np.array(env.target_grids) + env.M * env.N
target_id_states = env.target_grids + temp.tolist()

print "******************* Finish generating one day order **********************"



print "******************* Starting runing no policy baseline **********************"


MAX_ITER = 50  # 10 iteration the Q-learning loss will converge.
is_plot_figure = False
city_time_start = 0
EP_LEN = 144
global_step = 0
city_time_end = city_time_start + EP_LEN
epsilon = 0.5
gamma = 0.9
learning_rate = 1e-3

prev_epsiode_reward = 0
curr_num_actions = []
all_rewards = []
order_response_rate_episode = []
value_table_sum = []
episode_rewards = []
num_conflicts_drivers = []
driver_numbers_episode = []
order_numbers_episode = []

T = 144
action_dim = 7
state_dim = env.n_valid_grids * 3 + T

record_all_order_response_rate = []


def compute_context(target_grids, info):

    context = info.flatten()
    context = [context[idx] for idx in target_grids]
    return context

RATIO = 1

print "Start Running "
save_random_seed = []
episode_avaliables_vehicles = []
for n_iter in np.arange(10):
    RANDOM_SEED = n_iter + MAX_ITER + 5
    env.reset_randomseed(RANDOM_SEED)
    save_random_seed.append(RANDOM_SEED)
    batch_s, batch_a, batch_r = [], [], []
    batch_reward_gmv = []
    epsiode_reward = 0
    num_dispatched_drivers = 0

    driver_numbers = []
    order_numbers = []
    is_regenerate_order = 1
    curr_state = env.reset_clean(generate_order=is_regenerate_order, ratio=RATIO, city_time=city_time_start)
    driver_numbers.append(np.sum(curr_state[0]))
    order_numbers.append(np.sum(curr_state[1]))
    info = env.step_pre_order_assigin(curr_state)
    context = compute_context(env.target_grids, np.array(info))

    # record rewards to update the value table
    episodes_immediate_rewards = []
    order_response_rates = []
    available_drivers = []
    for ii in np.arange(EP_LEN + 1):
        available_drivers.append(np.sum(context))
        # ONE STEP: r0
        next_state, r, info = env.step([], 2)
        driver_numbers.append(np.sum(next_state[0]))
        order_numbers.append(np.sum(next_state[1]))

        context = compute_context(env.target_grids, np.array(info[1]))
        # Perform gradient descent update
        # book keeping
        global_step += 1
        all_rewards.append(r)
        batch_reward_gmv.append(r)
        order_response_rates.append(env.order_response_rate)

    episode_reward = np.sum(batch_reward_gmv[1:])
    episode_rewards.append(episode_reward)
    driver_numbers_episode.append(np.sum(driver_numbers[:-1]))
    order_numbers_episode.append(np.sum(order_numbers[:-1]))
    episode_avaliables_vehicles.append(np.sum(available_drivers[:-1]))
    n_iter_order_response_rate = np.mean(order_response_rates[1:])
    order_response_rate_episode.append(n_iter_order_response_rate)
    record_all_order_response_rate.append(order_response_rates)

    print "******** iteration {} ********* reward {}, order response rate {} available vehicle {}".format(n_iter,
                                                                                                          episode_reward,
                                                                                        n_iter_order_response_rate,
                                                                                        episode_avaliables_vehicles[-1])

    pickle.dump([episode_rewards, order_response_rate_episode, save_random_seed,
                 driver_numbers_episode, order_numbers_episode, episode_avaliables_vehicles], open(log_dir + "results.pkl", "w"))


print "averaged available vehicles per time step: {}".format(np.mean(episode_avaliables_vehicles)/144.0)