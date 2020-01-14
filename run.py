import json
import networkx as nx
import numpy as np
import os
import tarfile
import urllib.request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
UCI_ONLINE_URL = 'http://konect.uni-koblenz.de/downloads/tsv/opsahl-ucsocial.tar.bz2'
UCI_ONLINE_TAR_PATH = os.path.join(DATA_DIR, 'opsahl-ucsocial.tar.bz2')
UCI_ONLINE_DIR = os.path.join(DATA_DIR, 'opsahl-ucsocial')
UCI_ONLINE_TSV_PATH = os.path.join(UCI_ONLINE_DIR, 'out.opsahl-ucsocial')
TEMPORAL_BUCKET_SIZE = 3 * 60 * 60  # in seconds
DATA_SET_PATH = os.path.join(DATA_DIR, 'dataset.json')


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


# TODO gradient descent, genetic


def run():
    _ensure_data()
    data_set = _get_data_set()
    print(len(data_set))  # TODO remove


if __name__ == '__main__':
    run()
