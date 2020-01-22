import json
import math
import networkx as nx
import numpy as np
import os
import random
import tarfile
import urllib.request


# File Settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
UCI_ONLINE_URL = 'http://konect.uni-koblenz.de/downloads/tsv/opsahl-ucsocial.tar.bz2'
UCI_ONLINE_TAR_PATH = os.path.join(DATA_DIR, 'opsahl-ucsocial.tar.bz2')
UCI_ONLINE_DIR = os.path.join(DATA_DIR, 'opsahl-ucsocial')
UCI_ONLINE_TSV_PATH = os.path.join(UCI_ONLINE_DIR, 'out.opsahl-ucsocial')
TEMPORAL_BUCKET_SIZE = 3 * 60 * 60  # in seconds
DATA_SET_PATH = os.path.join(DATA_DIR, 'dataset.json')


# Gradient Descent Parameters
EPSILON = 10 ** -8
DATA_SET_SIZE = 10000
ITERATIONS = 100
ALPHA = 0.05


def _ensure_data():
    if not os.path.exists(UCI_ONLINE_DIR):
        urllib.request.urlretrieve(UCI_ONLINE_URL, UCI_ONLINE_TAR_PATH)
        tar = tarfile.open(UCI_ONLINE_TAR_PATH, "r:bz2")
        tar.extractall(DATA_DIR)
        tar.close()


def _entries_generator():
    with open(UCI_ONLINE_TSV_PATH, 'r') as tsv_file:
        for i, line in enumerate(tsv_file.readlines()):
            if not line.startswith('%'):
                split_line = line.strip().split()
                from_id = int(split_line[0])
                to_id = int(split_line[1])
                count = int(split_line[2])
                timestamp = int(split_line[3])
                yield from_id, to_id, count, timestamp


def _create_data_set():
    graph = nx.DiGraph()
    first_timestamp = 0
    last_timestamp = 0
    max_id = 0
    for from_id, to_id, count, timestamp in _entries_generator():
        graph.add_edge(from_id, to_id)
        if not first_timestamp:
            first_timestamp = timestamp
        last_timestamp = timestamp
        if from_id > max_id:
            max_id = from_id
        if to_id > max_id:
            max_id = to_id
    number_of_users = max_id + 1
    number_of_buckets = int((last_timestamp - first_timestamp) / TEMPORAL_BUCKET_SIZE) + 1
    activities = np.zeros((number_of_users, number_of_buckets))
    for from_id, to_id, count, timestamp in _entries_generator():
        bucket = int((timestamp - first_timestamp) / TEMPORAL_BUCKET_SIZE)
        activities[from_id, bucket] += count
    derivatives = np.diff(activities)
    data_set = []
    for node_id in range(1, number_of_users):
        neighbors = list(graph.neighbors(node_id))
        for bucket in range(number_of_buckets - 1):
            x = (activities[node_id, bucket], [activities[neighbor_node_id, bucket] for neighbor_node_id in neighbors])
            y = derivatives[node_id, bucket]
            data_set.append((x, y))
    return data_set


def _get_data_set():
    if os.path.exists(DATA_SET_PATH):
        with open(DATA_SET_PATH, 'r') as data_set_file:
            return json.loads(data_set_file.read())
    else:
        data_set = _create_data_set()
        with open(DATA_SET_PATH, 'w') as data_set_file:
            data_set_file.write(json.dumps(data_set))
        return data_set


def _gradient_descent(data_set):
    a1, a2, a3, a4, a5, a6, b1, b2, b3 = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    for i in range(ITERATIONS):
        print(a1, a2, a3, a4, a5, a6, b1, b2, b3)  # TODO remove

        dec_a1, dec_a2, dec_a3, dec_a4, dec_a5, dec_a6, dec_b1, dec_b2, dec_b3 = (0, 0, 0, 0, 0, 0, 0, 0, 0)

        mse = 0

        for x, y in data_set:
            # TODO problem: x_i may be negative, absolute is incorrect
            # TODO problem: if x_i is zero and power is negative
            x_i = x[0] + EPSILON
            x_js = [item + EPSILON for item in x[1]]

            dx_i = a1
            if x_i:
                dx_i += a2 * x_i ** b1
                for x_j in x[1]:
                    if x_j:
                        dx_i += (a3 + a4 * x_i ** b2) * (a5 + a6 * x_j ** b3)
            mse += (y - dx_i) ** 2

            dec_a1 += ALPHA * (y - dx_i) * 1
            dec_a2 += ALPHA * (y - dx_i) * (x_i ** b1)
            dec_a3 += ALPHA * (y - dx_i) * sum([(a5 + a6 * (x_j ** b3)) for x_j in x_js])
            dec_a4 += ALPHA * (y - dx_i) * sum([(a5 * (x_i ** b2) + a6 * (x_i ** b2) * (x_j ** b3)) for x_j in x_js])
            dec_a5 += ALPHA * (y - dx_i) * sum([(a3 + a4 * x_j ** b2) for x_j in x_js])
            dec_a6 += ALPHA * (y - dx_i) * sum([(a3 * (x_j ** b3) + a4 * (x_i ** b2) * (x_j ** b3)) for x_j in x_js])
            dec_b1 += ALPHA * (y - dx_i) * (a2 * math.log(abs(x_i)) * (x_i ** b1))
            dec_b2 += ALPHA * (y - dx_i) * \
                  sum([(a4 * (a5 + a6 * (x_j ** b3)) * math.log(abs(x_i)) * (x_i ** b2)) for x_j in x_js])
            dec_b3 += ALPHA * (y - dx_i) * \
                  sum([(a6 * (a3 + a4 * (x_i ** b2)) * math.log(abs(x_j)) * (x_j ** b3)) for x_j in x_js])

        mse /= len(data_set)
        print(mse)

        a1 -= dec_a1
        a2 -= dec_a2
        a3 -= dec_a3
        a4 -= dec_a4
        a5 -= dec_a5
        a6 -= dec_a6
        b1 -= dec_b1
        b2 -= dec_b2
        b3 -= dec_b3

    print(a1, a2, a3, a4, a5, a6, b1, b2, b3)


def _get_balanced_data_set(data_set, data_set_size):
    non_zero_data_set = [item for item in data_set if item[1]]
    zero_data_set = [item for item in data_set if not item[1]]
    new_data_set = non_zero_data_set[:int(data_set_size / 2)]
    new_data_set += zero_data_set[:data_set_size - len(new_data_set)]
    random.shuffle(new_data_set)
    return new_data_set


def run():
    _ensure_data()
    balanced_data_set = _get_balanced_data_set(_get_data_set(), DATA_SET_SIZE)
    _gradient_descent(balanced_data_set)


if __name__ == '__main__':
    run()
