import json
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


# Genetic Settings
DATA_SET_SIZE = 10000
POPULATION = 100
GENERATIONS = 1000
MUTATION_CHANCE = 0.05


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


def _calculate_derivative(genes, x):
    a1, a2, a3, a4, a5, a6, b1, b2, b3 = genes
    x_i = x[0]
    result = a1
    if x_i:
        result += a2 * x_i ** b1
        for x_j in x[1]:
            if x_j:
                result += (a3 + a4 * x_i ** b2) * (a5 + a6 * x_j ** b3)
    return result


def _calculate_error(genes, data_set):
    mse = 0
    for x, y in data_set:
        mse += (y - _calculate_derivative(genes, x)) ** 2
    mse /= len(data_set)
    return mse


def _get_random_gene():
    return (random.random() - 0.5) * 2


def _crossover(parent1, parent2):
    new_chromosome = []
    crossover_point = random.randint(1, len(parent1))
    for i in range(crossover_point):
        new_chromosome.append(parent1[i])
    for i in range(crossover_point, len(parent1)):
        new_chromosome.append(parent2[i])
    return new_chromosome


def _mutation(chromosome):
    new_chromosome = []
    for gene in chromosome:
        if random.random() < MUTATION_CHANCE:
            new_chromosome.append(gene + _get_random_gene())  # if we don't add, we never get powers outside (-1, 1)
        else:
            new_chromosome.append(gene)
    return new_chromosome


def _genetic(data_set):
    population = [[_get_random_gene() for j in range(9)] for i in range(POPULATION)]
    for i in range(GENERATIONS):
        fitness_values = [1 / _calculate_error(chromosome, data_set) for chromosome in population]
        print(max(fitness_values))
        fitness_sum = sum(fitness_values)
        weights = [fitness / fitness_sum for fitness in fitness_values]
        new_population = []
        for j in range(POPULATION):
            parent1 = random.choices(population=population, weights=weights)[0]
            parent2 = random.choices(population=population, weights=weights)[0]
            new_chromosome = _mutation(_crossover(parent1, parent2))
            new_population.append(new_chromosome)
        population = new_population
    fitness_values = [1 / _calculate_error(chromosome, data_set) for chromosome in population]
    fittest_index = fitness_values.index(max(fitness_values))
    print(population[fittest_index])


def _get_balanced_data_set(data_set, data_set_size):
    non_zero_data_set = [item for item in data_set if item[1]]
    zero_data_set = [item for item in data_set if not item[1]]
    new_data_set = non_zero_data_set[:int(data_set_size / 2)]
    new_data_set += zero_data_set[:data_set_size - len(new_data_set)]
    random.shuffle(new_data_set)
    return new_data_set


# TODO gradient descent + validation
def run():
    _ensure_data()
    balanced_data_set = _get_balanced_data_set(_get_data_set(), DATA_SET_SIZE)
    _genetic(balanced_data_set)


if __name__ == '__main__':
    run()
