
from collections import defaultdict
import sys
import traceback
import os, sys
sys.path.append("../")
from simulator.envs import *


def running_example():
    mapped_matrix_int = np.array([[1, -100, 3], [5, 4, 2], [6, 7, 8]])
    M, N = mapped_matrix_int.shape
    order_num_dist = []
    num_valid_grid = 8
    idle_driver_location_mat = np.zeros((144, 8))

    for ii in np.arange(144):
        time_dict = {}
        for jj in np.arange(M*N):  # num of grids
            time_dict[jj] = [2]
        order_num_dist.append(time_dict)
        idle_driver_location_mat[ii, :] = [2] * num_valid_grid

    idle_driver_dist_time = [[10, 1] for _ in np.arange(144)]

    n_side = 6
    l_max = 2
    order_time = [0.2, 0.2, 0.15,
                  0.15,  0.1,  0.1,
                  0.05, 0.04,  0.01]
    order_price = [[10.17, 3.34],  # mean and std of order price when duration is 10 min
                   [15.02, 6.90],  # mean and std of order price when duration is 20 min
                   [23.22, 11.63],
                   [32.14, 16.20],
                   [40.99, 20.69],
                   [49.94, 25.61],
                   [58.98, 31.69],
                   [68.80, 37.25],
                   [79.40, 44.39]]

    order_real = []
    onoff_driver_location_mat = []
    for tt in np.arange(144):
        order_real += [[1, 5, tt, 1, 13.2], [2, 6, tt, 1, 13.2], [3, 1, tt, 1, 13.2],
                       [5, 3, tt, 1, 13.2], [6, 2, tt, 1, 13.2], [7, 9, tt, 1, 13.2],
                       [9, 10, tt, 1, 13.2], [10, 6, tt, 1, 13.2], [9, 7, tt, 1, 13.2]]
        onoff_driver_location_mat.append([[-0.625,       2.92350389],
                                         [0.09090909,  1.46398452],
                                         [0.09090909,  2.36596622],
                                         [0.09090909, 2.36596622],
                                         [0.09090909, 1.46398452],
                                         [0.09090909, 1.46398452],
                                         [0.09090909, 1.46398452],
                                         [0.09090909, 1.46398452],
                                         [0.09090909, 1.46398452]])
    env = CityReal(mapped_matrix_int, order_num_dist, idle_driver_dist_time, idle_driver_location_mat,
                   order_time, order_price, l_max, M, N, n_side, 1, np.array(order_real), np.array(onoff_driver_location_mat))

    state = env.reset_clean()
    order_response_rates = []
    T = 0
    max_iter = 144
    while T < max_iter:
        # if T % 5 == 0:
        #     state = env.reset_clean(generate_order=2)
        dispatch_action = []
        state, reward, _ = env.step(dispatch_action, generate_order=2)

        print "City time {}: Order response rate: {}".format(env.city_time-1, env.order_response_rate)
        order_response_rates.append(env.order_response_rate)

        print("idle driver: {} == {} total num of drivers: {}".format(np.sum(state[0]),
                                                                      np.sum(env.get_observation_driver_state()),
                                                                      len(env.drivers.keys())))

        assert np.sum(state[0]) == env.n_drivers

        T += 1
    print np.mean(order_response_rates)


if __name__ == "__main__":
    running_example()

