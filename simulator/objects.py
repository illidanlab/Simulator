import numpy as np
from abc import ABCMeta, abstractmethod
from utilities import *

class Distribution():
    ''' Define the distribution from which sample the orders'''
    __metaclass__ = ABCMeta  # python 2.7
    @abstractmethod
    def sample(self):
        pass

class PoissonDistribution(Distribution):

    def __init__(self, lam):
        self._lambda = lam

    def sample(self, seed=0):
        np.random.seed(seed)
        return np.random.poisson(self._lambda, 1)[0]


class GaussianDistribution(Distribution):

    def __init__(self, args):
        mu, sigma = args
        self.mu = mu        # mean
        self.sigma = sigma  # standard deviation

    def sample(self, seed=0):
        np.random.seed(seed)
        return np.random.normal(self.mu, self.sigma, 1)[0]


class Node(object):
    __slots__ = ('neighbors', '_index', 'orders', 'drivers',
                 'order_num', 'idle_driver_num', 'offline_driver_num'
                 'order_generator', 'offline_driver_num', 'order_generator',
                 'n_side', 'layers_neighbors', 'layers_neighbors_id')

    def __init__(self, index):
        # private
        self._index = index   # unique node index.

        # public
        self.neighbors = []  # a list of nodes that neighboring the Nodes
        self.orders = []     # a list of orders
        self.drivers = {}    # a dictionary of driver objects contained in this node
        self.order_num = 0
        self.idle_driver_num = 0  # number of idle drivers in this node
        self.offline_driver_num = 0
        self.order_generator = None

        self.n_side = 0      # the topology is a n-sided map
        self.layers_neighbors = []  # layer 1 indices: layers_neighbors[0] = [[1,1], [0, 1], ...],
        # layer 2 indices layers_neighbors[1]
        self.layers_neighbors_id = [] # layer 1: layers_neighbors_id[0] = [2, 1,.]

    def clean_node(self):
        self.orders = []
        self.order_num = 0
        self.drivers = {}
        self.idle_driver_num = 0
        self.offline_driver_num = 0

    def get_layers_neighbors(self, l_max, M, N, env):

        x, y = ids_1dto2d(self.get_node_index(), M, N)
        self.layers_neighbors = get_layers_neighbors(x, y, l_max, M, N)
        for layer_neighbors in self.layers_neighbors:
            temp = []
            for item in layer_neighbors:
                x, y = item
                node_id = ids_2dto1d(x, y, M, N)
                if env.nodes[node_id] is not None:
                    temp.append(node_id)
            self.layers_neighbors_id.append(temp)

    def get_node_index(self):
        return self._index

    def get_driver_numbers(self):
        return self.idle_driver_num

    def get_idle_driver_numbers_loop(self):
        temp_idle_driver = 0
        for key, driver in self.drivers.iteritems():
            if driver.onservice is False and driver.online is True:
                temp_idle_driver += 1
        return temp_idle_driver

    def get_off_driver_numbers_loop(self):
        temp_idle_driver = 0
        for key, driver in self.drivers.iteritems():
            if driver.onservice is False and driver.online is False:
                temp_idle_driver += 1
        return temp_idle_driver

    def order_distribution(self, distribution, dis_paras):

        if distribution == 'Poisson':
            self.order_generator = PoissonDistribution(dis_paras)
        elif distribution == 'Gaussian':
            self.order_generator = GaussianDistribution(dis_paras)
        else:
            pass

    def generate_order_random(self, city_time, nodes, seed):
        """Generate new orders at each time step
        """
        num_order_t = self.order_generator.sample(seed)
        self.order_num += num_order_t

        for ii in np.arange(num_order_t):
            price = np.random.normal(50, 5, 1)[0]
            price = 10 if price < 0 else price

            current_node_id = self.get_node_index()
            destination_node = [kk for kk in np.arange(len(nodes)) if kk != current_node_id]
            self.orders.append(Order(nodes[current_node_id],
                                     nodes[np.random.choice(destination_node, 1)[0]],
                                     city_time,
                                     # city_time + np.random.choice(5, 1)[0]+1,
                                     np.random.choice(2, 1)[0]+1,  # duration
                                     price, 1))
        return

    def generate_order_real(self, l_max, order_time_dist, order_price_dist, city_time, nodes, seed):
        """Generate new orders at each time step
        """
        num_order_t = self.order_generator.sample(seed)
        self.order_num += num_order_t
        for ii in np.arange(num_order_t):

            if l_max == 1:
                duration = 1
            else:

                duration = np.random.choice(np.arange(1, l_max+1), p=order_time_dist)
            price_mean, price_std = order_price_dist[duration-1]
            price = np.random.normal(price_mean, price_std, 1)[0]
            price = price if price > 0 else price_mean

            current_node_id = self.get_node_index()
            destination_node = []
            for jj in np.arange(duration):
                for kk in self.layers_neighbors_id[jj]:
                    if nodes[kk] is not None:
                        destination_node.append(kk)
            self.orders.append(Order(nodes[current_node_id],
                                     nodes[np.random.choice(destination_node, 1)[0]],
                                     city_time,
                                     duration,
                                     price, 1))
        return

    def add_order_real(self, city_time, destination_node, duration, price):
        current_node_id = self.get_node_index()
        self.orders.append(Order(self,
                                 destination_node,
                                 city_time,
                                 duration,
                                 price, 0))
        self.order_num += 1

    def set_neighbors(self, nodes_list):
        self.neighbors = nodes_list
        self.n_side = len(nodes_list)

    def remove_idle_driver_random(self):
        """Randomly remove one idle driver from current grid"""
        removed_driver_id = "NA"
        for key, item in self.drivers.iteritems():
            if item.onservice is False and item.online is True:
                self.remove_driver(key)
                removed_driver_id = key
            if removed_driver_id != "NA":
                break
        assert removed_driver_id != "NA"
        return removed_driver_id

    def set_idle_driver_offline_random(self):
        """Randomly set one idle driver offline"""
        removed_driver_id = "NA"
        for key, item in self.drivers.iteritems():
            if item.onservice is False and item.online is True:
                item.set_offline()
                removed_driver_id = key
            if removed_driver_id != "NA":
                break
        assert removed_driver_id != "NA"
        return removed_driver_id

    def set_offline_driver_online(self):

        online_driver_id = "NA"
        for key, item in self.drivers.iteritems():
            if item.onservice is False and item.online is False:
                item.set_online()
                online_driver_id = key
            if online_driver_id != "NA":
                break
        assert online_driver_id != "NA"
        return online_driver_id

    def get_driver_random(self):
        """Randomly get one driver"""
        assert self.idle_driver_num > 0
        get_driver_id = 0
        for key in self.drivers.iterkeys():
            get_driver_id = key
            break
        return self.drivers[get_driver_id]

    def remove_driver(self, driver_id):

        removed_driver = self.drivers.pop(driver_id, None)
        self.idle_driver_num -= 1
        if removed_driver is None:
            raise ValueError('Nodes.remove_driver: Remove a driver that is not in this node')

        return removed_driver

    def add_driver(self, driver_id, driver):
        self.drivers[driver_id] = driver
        self.idle_driver_num += 1

    def remove_unfinished_order(self, city_time):
        un_finished_order_index = []
        for idx, o in enumerate(self.orders):
            # order un served
            if o.get_wait_time()+o.get_begin_time() < city_time:
                un_finished_order_index.append(idx)

            # order completed
            if o.get_assigned_time() + o.get_duration() == city_time and o.get_assigned_time() != -1:
                un_finished_order_index.append(idx)

        if len(un_finished_order_index) != 0:
            # remove unfinished orders
            self.orders = [i for j, i in enumerate(self.orders) if j not in un_finished_order_index]
            self.order_num = len(self.orders)

    def simple_order_assign(self, city_time, city):
        reward = 0
        num_assigned_order = min(self.order_num, self.idle_driver_num)
        served_order_index = []
        for idx in np.arange(num_assigned_order):
            order_to_serve = self.orders[idx]
            order_to_serve.set_assigned_time(city_time)
            self.order_num -= 1
            reward += order_to_serve.get_price()
            served_order_index.append(idx)
            for key, assigned_driver in self.drivers.iteritems():
                if assigned_driver.onservice is False and assigned_driver.online is True:
                    assigned_driver.take_order(order_to_serve)
                    removed_driver = self.drivers.pop(assigned_driver.get_driver_id(), None)
                    assert removed_driver is not None
                    city.n_drivers -= 1
                    break

        all_order_num = len(self.orders)
        finished_order_num = len(served_order_index)

        # remove served orders
        self.orders = [i for j, i in enumerate(self.orders) if j not in served_order_index]
        assert self.order_num == len(self.orders)

        return reward, all_order_num, finished_order_num

    def simple_order_assign_real(self, city_time, city):

        reward = 0
        num_assigned_order = min(self.order_num, self.idle_driver_num)
        served_order_index = []
        for idx in np.arange(num_assigned_order):
            order_to_serve = self.orders[idx]
            order_to_serve.set_assigned_time(city_time)
            self.order_num -= 1
            reward += order_to_serve.get_price()
            served_order_index.append(idx)
            for key, assigned_driver in self.drivers.iteritems():
                if assigned_driver.onservice is False and assigned_driver.online is True:
                    if order_to_serve.get_end_position() is not None:
                        assigned_driver.take_order(order_to_serve)
                        removed_driver = self.drivers.pop(assigned_driver.get_driver_id(), None)
                        assert removed_driver is not None
                    else:
                        assigned_driver.set_offline()  # order destination is not in target region
                    city.n_drivers -= 1
                    break

        all_order_num = len(self.orders)
        finished_order_num = len(served_order_index)

        # remove served orders
        self.orders = [i for j, i in enumerate(self.orders) if j not in served_order_index]
        assert self.order_num == len(self.orders)

        return reward, all_order_num, finished_order_num


    def simple_order_assign_broadcast_update(self, city, neighbor_node_reward):

        assert self.idle_driver_num == 0
        reward = 0
        num_finished_orders = 0
        for neighbor_node in self.neighbors:
            if neighbor_node is not None and neighbor_node.idle_driver_num > 0:
                num_assigned_order = min(self.order_num, neighbor_node.idle_driver_num)
                rr = self.utility_assign_orders_neighbor(city, neighbor_node, num_assigned_order)
                reward += rr
                neighbor_node_reward[neighbor_node.get_node_index()] += rr
                num_finished_orders += num_assigned_order
            if self.order_num == 0:
                break

        assert self.order_num == len(self.orders)
        return reward, num_finished_orders

    def utility_assign_orders_neighbor(self, city, neighbor_node, num_assigned_order):

        served_order_index = []
        reward = 0
        curr_city_time = city.city_time
        for idx in np.arange(num_assigned_order):
            order_to_serve = self.orders[idx]
            order_to_serve.set_assigned_time(curr_city_time)
            self.order_num -= 1
            reward += order_to_serve.get_price()
            served_order_index.append(idx)
            for key, assigned_driver in neighbor_node.drivers.iteritems():
                if assigned_driver.onservice is False and assigned_driver.online is True:
                    if order_to_serve.get_end_position() is not None:
                        assigned_driver.take_order(order_to_serve)
                        removed_driver = neighbor_node.drivers.pop(assigned_driver.get_driver_id(), None)
                        assert removed_driver is not None
                    else:
                        assigned_driver.set_offline()
                    city.n_drivers -= 1
                    break

        # remove served orders
        self.orders = [i for j, i in enumerate(self.orders) if j not in served_order_index]
        assert self.order_num == len(self.orders)

        return reward


class Driver(object):
    __slots__ = ("online", "onservice", 'order', 'node', 'city_time', '_driver_id')

    def __init__(self, driver_id):
        self.online = True
        self.onservice = False
        self.order = None     # the order this driver is serving
        self.node = None      # the node that contain this driver.
        self.city_time = 0  # track the current system time

        # private
        self._driver_id = driver_id  # unique driver id.

    def set_position(self, node):
        self.node = node

    def set_order_start(self, order):
        self.order = order

    def set_order_finish(self):
        self.order = None
        self.onservice = False

    def get_driver_id(self):
        return self._driver_id

    def update_city_time(self):
        self.city_time += 1

    def set_city_time(self, city_time):
        self.city_time = city_time

    def set_offline(self):
        assert self.onservice is False and self.online is True
        self.online = False
        self.node.idle_driver_num -= 1
        self.node.offline_driver_num += 1

    def set_offline_for_start_dispatch(self):

        assert self.onservice is False
        self.online = False

    def set_online(self):
        assert self.onservice is False
        self.online = True
        self.node.idle_driver_num += 1
        self.node.offline_driver_num -= 1

    def set_online_for_finish_dispatch(self):

        self.online = True
        assert self.onservice is False

    def take_order(self, order):
        """ take order, driver show up at destination when order is finished
        """
        assert self.online == True
        self.set_order_start(order)
        self.onservice = True
        self.node.idle_driver_num -= 1

    def status_control_eachtime(self, city):

        assert self.city_time == city.city_time
        if self.onservice is True:
            assert self.online is True
            order_end_time = self.order.get_assigned_time() + self.order.get_duration()
            if self.city_time == order_end_time:
                self.set_position(self.order.get_end_position())
                self.set_order_finish()
                self.node.add_driver(self._driver_id, self)
                city.n_drivers += 1
            elif self.city_time < order_end_time:
                pass
            else:
                raise ValueError('Driver: status_control_eachtime(): order end time less than city time')


class Order(object):
    __slots__ = ('_begin_p', '_end_p', '_begin_t',
                 '_t', '_p', '_waiting_time', '_assigned_time')

    def __init__(self, begin_position, end_position, begin_time, duration, price, wait_time):
        self._begin_p = begin_position  # node
        self._end_p = end_position      # node
        self._begin_t = begin_time
        # self._end_t = end_time
        self._t = duration              # the duration of order.
        self._p = price
        self._waiting_time = wait_time  # a order can last for "wait_time" to be taken
        self._assigned_time = -1

    def get_begin_position(self):
        return self._begin_p

    def get_begin_position_id(self):
        return self._begin_p.get_node_index()

    def get_end_position(self):
        return self._end_p

    def get_begin_time(self):
        return self._begin_t

    def set_assigned_time(self, city_time):
        self._assigned_time = city_time

    def get_assigned_time(self):
        return self._assigned_time

    # def get_end_time(self):
    #     return self._end_t

    def get_duration(self):
        return self._t

    def get_price(self):
        return self._p

    def get_wait_time(self):
        return self._waiting_time
