import os
import math
import datetime
import sys
import random
import copy

import numpy as np

import consequence.conf as conf
from consequence.conf import Conf, Model, TOP_K, TIME_BUDGET, THETA, ITERATIONS_LIMIT, MAX_LENGTH, MAX_GAP, SUPPORT_PENALTY, MIN_SUPPORT
from utils.reader import read_data_from_csv
from consequence.utils import sequence_mutable_to_immutable, compute_quality, \
    sequence_immutable_to_mutable, filter_empty_sequences, encode_items, \
    encode_data, print_results_decode, extract_items, decode_sequences

from consequence.priorityset import PrioritySet, ExperimentContext
from consequence.node import Node

sys.setrecursionlimit(15000)


class LoadedDataset:
    def __init__(self, data, target_class, items_to_encoding, encoding_to_items):
        self.data = data
        self.target_class = target_class
        self.items_to_encoding = items_to_encoding
        self.encoding_to_items = encoding_to_items


def resolve_dataset_csv(dataset):
    if os.path.isfile(dataset):
        return dataset

    candidates = [
        f"data/{dataset}_train.csv",
        f"data/{dataset}.csv",
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(f"Could not find CSV dataset for {dataset!r}")


def load_encoded_dataset(csv_path):
    data, target_class = read_data_from_csv(csv_path)
    items = extract_items(data)
    items, items_to_encoding, encoding_to_items = encode_items(items)
    data = encode_data(data, items_to_encoding)

    return LoadedDataset(
        data=data,
        target_class=target_class,
        items_to_encoding=items_to_encoding,
        encoding_to_items=encoding_to_items,
    )


def setup_model_targets(model, target_class):
    labels = sorted(np.unique(target_class[:, 0]))
    positive_class = labels[1] if len(labels) == 2 else labels[-1]

    model.labels = list(labels)
    model.positive_class = positive_class
    model.target_class = target_class

    confidences = target_class[:, 1]
    y_trues = target_class[:, 0]
    p = np.where(y_trues == positive_class, confidences, 1.0 - confidences)
    soft_errors = np.abs(1.0 - p)
    predictions = np.where(confidences > 0.5, positive_class, labels[0])
    hard_errors = (y_trues != predictions).astype(float)

    model.soft_errors = soft_errors
    model.hard_errors = hard_errors
    model.global_errors = soft_errors
    model.global_mean_error = float(soft_errors.mean())
    model.global_std_error = float(soft_errors.std())
    model.global_hard_error = float(hard_errors.mean())

    print("\n=== Precomputed Metrics ===")
    print(f"Global hard error rate: {hard_errors.mean():.4f}")
    print(f"Global soft mean error: {soft_errors.mean():.4f}")


def best_child(node):
    if node.is_dead_end() and len(node.parents) == 0:
        return 'finished'

    best_node = None
    max_score = -float("inf")

    for child in node.children:
        if child.is_dead_end():
            continue
        a = child.get_normalized_quality() / child.number_visits
        b = 0.5 * math.sqrt(2 * math.log(node.number_visits) / child.number_visits)
        current_ucb = a + b

        if current_ucb > max_score:
            max_score = current_ucb
            best_node = child

    if best_node is None:
        return node.parents[0]

    return best_node


def select(node):
    while node != 'finished':
        if len(node.children) == 0:
            return (node, False)
        if (random.random() < 0.5) and (not node.is_fully_expanded()):
            return (node, True)
        node = best_child(node)
    return ('finished', None)


def roll_out(node):
    sequence = copy.deepcopy(node.intent)
    sequence = sequence_immutable_to_mutable(sequence)

    if not sequence or len(sequence) <= 1:
        return sequence, 1, 0, [], 0, 0

    num_to_remove = random.randint(
        max(1, int(len(sequence) * 0.2)),
        max(1, int(len(sequence) * 0.5))
    )

    choice = random.randint(0, 2)

    if choice == 0:
        sequence = sequence[num_to_remove:]
    elif choice == 1:
        sequence = sequence[:-num_to_remove]
    else:
        remove_start = num_to_remove // 2
        remove_end = num_to_remove - remove_start
        sequence = sequence[remove_start:-remove_end if remove_end > 0 else None]

    if not sequence:
        return [], 0, 0, [], 0, 0

    immutable_sequence = tuple(sequence_mutable_to_immutable(sequence))
    quality, error_delta, extend, size_class_0, size_class_1 = compute_quality(immutable_sequence)

    return immutable_sequence, quality, error_delta, extend, size_class_0, size_class_1


def update(node, reward):
    update_nodes = {node}
    parents_seen = set()

    while len(update_nodes) != 0:
        node = random.sample(update_nodes, 1)[0]
        parents_seen.add(node)
        for parent in node.parents:
            if parent not in parents_seen:
                update_nodes.add(parent)

        node.update(reward)
        update_nodes.remove(node)


def get_patterns(
    dataset='',
    top_k=TOP_K,
    time_budget=TIME_BUDGET,
    theta=THETA,
    iterations_limit=ITERATIONS_LIMIT,
    max_length=MAX_LENGTH,
    max_gap=MAX_GAP,
    support_penalty=SUPPORT_PENALTY,
    min_support=MIN_SUPPORT,
):
    conf.conf = Conf(
        top_k=top_k,
        time_budget=time_budget,
        theta=theta,
        iterations_limit=iterations_limit,
        max_length=max_length,
        max_gap=max_gap,
        support_penalty=support_penalty,
        min_support=min_support,
    )
    conf.model = Model()

    train_csv = f"data/{dataset}_train.csv"
    loaded = load_encoded_dataset(train_csv)
    setup_model_targets(conf.model, loaded.target_class)

    dataset_name = dataset if not os.path.isfile(dataset) else os.path.splitext(os.path.basename(dataset))[0]
    if dataset_name.endswith('_train'):
        dataset_name = dataset_name[:-6]

    context = ExperimentContext(
        dataset_name=dataset_name,
        paths=train_csv,
        items_to_encoding=loaded.items_to_encoding,
    )

    results = launch_mcts(loaded.data, loaded.target_class, context)
    print_results_decode(results, loaded.encoding_to_items)
    return decode_sequences(results, loaded.encoding_to_items)


def extend_cover_minsup_abs(extend):
    return len(extend) >= conf.conf.min_support


def build_iteration_metrics(node_hashmap, iteration_count, runtime_seconds, sorted_patterns):
    return {
        'nodes_visited': len(node_hashmap),
        'iterations': iteration_count,
        'runtime_seconds': runtime_seconds,
        'unique_patterns_in_queue': len(sorted_patterns.set),
    }


def launch_mcts(data, target_class, context):
    settings = conf.conf
    model = conf.model
    begin = datetime.datetime.utcnow()
    deadline = begin + datetime.timedelta(seconds=settings.time_budget)

    data = filter_empty_sequences(data)
    model.target_class = target_class
    model.data = data

    node_hashmap = {}
    root_node = Node(None, None, node_hashmap)
    node_hashmap[('.')] = root_node

    sorted_patterns = PrioritySet(k=settings.top_k, theta=settings.theta)
    iteration_count = 0
    highest_error = 0
    min_per_class = 10

    while datetime.datetime.utcnow() <= deadline and iteration_count < settings.iterations_limit:
        node_sel, _ = select(root_node)

        if node_sel == 'finished':
            print('Finished')
            break

        node_expand = node_sel.expand()

        if (node_expand.quality > 0 and node_expand.error_delta > 0 and len(node_expand.intent)
                and extend_cover_minsup_abs(node_expand.extend)
                and node_expand.size_class_0 >= min_per_class and node_expand.size_class_1 >= min_per_class):
            sorted_patterns.add(
                sequence_mutable_to_immutable(node_expand.intent),
                node_expand.quality,
                node_expand.extend,
                node_expand.error_delta,
            )
            if node_expand.error_delta > highest_error:
                highest_error = node_expand.error_delta
                print(f"Highest Error: {highest_error}. Pattern: {node_expand.intent}")

        reward_intent, reward, reward_error_delta, reward_extend, reward_n0, reward_n1 = roll_out(node_expand)

        if (reward > 0 and reward_error_delta > 0 and len(reward_intent)
                and extend_cover_minsup_abs(reward_extend)
                and reward_n0 >= min_per_class and reward_n1 >= min_per_class):
            sorted_patterns.add(reward_intent, reward, reward_extend, reward_error_delta)
            if reward_error_delta > highest_error:
                highest_error = reward_error_delta
                print(f"Highest Error: {highest_error}. Pattern: {reward_intent}")

        update(node_expand, reward)
        iteration_count += 1

        if iteration_count % 100 == 0:
            print(iteration_count)

    print(f'Number iteration mcts: {iteration_count}')
    context.iteration_count = iteration_count
    context.iteration_metrics = build_iteration_metrics(
        node_hashmap,
        iteration_count,
        (datetime.datetime.utcnow() - begin).total_seconds(),
        sorted_patterns,
    )
    return sorted_patterns.get_top_k_non_redundant(
        data,
        settings.top_k,
        context=context,
    )
