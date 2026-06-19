import heapq
import numpy as np
from datetime import datetime

from consequence.utils import decode_sequence

from utils.statistical_validation import filter_by_significance
from utils.save_results import save_all_patterns, save_patterns_after_similarity_filter, save_patterns_after_stats_validation, save_iteration_metrics
import consequence.conf as conf


class ExperimentContext:
    def __init__(
        self,
        dataset_name,
        paths,
        items_to_encoding,
        iteration_count=None,
        iteration_metrics=None,
    ):
        self.dataset_name = dataset_name
        self.paths = paths
        self.items_to_encoding = items_to_encoding
        self.iteration_count = iteration_count
        self.iteration_metrics = iteration_metrics


def jaccard_from_extends(extend1, extend2):
    if not extend1 and not extend2:
        return 0.0
    s1 = extend1 if isinstance(extend1, set) else set(extend1)
    s2 = extend2 if isinstance(extend2, set) else set(extend2)
    intersection = len(s1 & s2)
    if intersection == 0:
        return 0.0
    union = len(s1 | s2)
    return intersection / union if union else 0.0


def decode_results(results_list, items_to_encoding):
    encoding_to_items = {v: k for k, v in items_to_encoding.items()} if items_to_encoding else None
    decoded_results = []

    target_class = conf.model.target_class
    soft_errors = conf.model.soft_errors

    for idx, result in enumerate(results_list):
        quality, sequence, extend, error_delta = result
        pattern_display = ''
        decoded_seq = decode_sequence(sequence, encoding_to_items)
        for itemset in decoded_seq:
            pattern_display += repr(set(itemset))

        error_class_0 = None
        error_class_1 = None
        std_class_0 = None
        std_class_1 = None
        size_class_0 = 0
        size_class_1 = 0
        
        if target_class is not None and soft_errors is not None and len(extend) > 0:
            extend_arr = np.array(extend, dtype=int)
            y_true = target_class[:, 0]
            
            subgroup_class_0 = soft_errors[extend_arr][y_true[extend_arr] == 0]
            subgroup_class_1 = soft_errors[extend_arr][y_true[extend_arr] == 1]
            
            error_class_0 = subgroup_class_0.mean() if len(subgroup_class_0) > 0 else 0.0
            error_class_1 = subgroup_class_1.mean() if len(subgroup_class_1) > 0 else 0.0
            std_class_0 = float(subgroup_class_0.std()) if len(subgroup_class_0) >= 2 else 0.0
            std_class_1 = float(subgroup_class_1.std()) if len(subgroup_class_1) >= 2 else 0.0
            size_class_0 = len(subgroup_class_0)
            size_class_1 = len(subgroup_class_1)

        result_dict = { 
            'pattern': decoded_seq, 
            'quality': quality, 
            'support': len(extend), 
            'error_delta': error_delta 
        }
        
        if error_class_0 is not None and error_class_1 is not None:
            result_dict['error_class_0'] = error_class_0
            result_dict['error_class_1'] = error_class_1
            result_dict['size_class_0'] = size_class_0
            result_dict['size_class_1'] = size_class_1
            result_dict['std_class_0'] = std_class_0
            result_dict['std_class_1'] = std_class_1
        
        decoded_results.append(result_dict)
        
        if error_class_0 is not None and error_class_1 is not None:
            print(f"  Pattern {idx}: Quality={quality:.4f}, Error_Delta={error_delta:.4f}, Support={len(extend)}, Class0_Error={error_class_0:.4f} (n={size_class_0}), Class1_Error={error_class_1:.4f} (n={size_class_1}), Class0_Std={std_class_0:.4f}, Class1_Std={std_class_1:.4f}, Pattern={pattern_display}")
        else:
            print(f"  Pattern {idx}: Quality={quality:.4f}, Error_Delta={error_delta:.4f}, Support={len(extend)}, Pattern={pattern_display}")
    print(f"{'='*80}\n")
    return decoded_results


def filter_results(results, data, theta, k, k_prime=100, alpha=0.05, context=None):
    settings = conf.conf
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_list = list(results)
    results_list.sort(key=lambda x: x[0], reverse=True)
    results_list = [result for result in results_list if len(result[1]) <= settings.max_length]

    print(f"================\nALL PATTERNS\n================")
    d_results = decode_results(results_list, context.items_to_encoding)

    save_all_patterns(
        d_results,
        len(data),
        context.dataset_name,
        timestamp,
        context.iteration_count,
        max_gap=settings.max_gap,
        support_penalty=settings.support_penalty,
    )

    save_iteration_metrics(
        context.iteration_metrics,
        context.dataset_name,
        timestamp,
        context.iteration_count,
    )

    print(f"================\nFILTERING BY SIMILARITY\n================")
    non_redundant_patterns = []
    non_redundant_extends = []
    for idx, result in enumerate(results_list):
        if idx and idx % 100 == 0:
            print(f"Similarity filter: {idx}/{len(results_list)} patterns checked")
        extend = set(result[2])
        similar = False
        for kept_extend in non_redundant_extends:
            if jaccard_from_extends(extend, kept_extend) > theta:
                similar = True
                break
        if not similar:
            non_redundant_patterns.append(result)
            non_redundant_extends.append(extend)
        if len(non_redundant_patterns) == 100:
            break

    d_results = decode_results(non_redundant_patterns, context.items_to_encoding)
    save_patterns_after_similarity_filter(d_results, theta, context.dataset_name, timestamp, context.iteration_count)

    print(f"================\nAPPLYING STATISTICAL VALIDATION\n================")
    significant_patterns, significance_info = filter_by_significance(
        non_redundant_patterns[:k_prime],
        validation_csv=context.paths.validation_csv,
        train_csv=context.paths.train_csv,
        items_to_encoding=context.items_to_encoding,
        alpha=alpha,
        n_subgroups=1000,
    )

    print(f"================\nPATTERNS AFTER STATISTICAL VALIDATION\n================")
    decode_results(significant_patterns, context.items_to_encoding)

    save_patterns_after_stats_validation(significance_info, context.dataset_name, timestamp, context.iteration_count)

    return significant_patterns[:k]


class PrioritySet(object):
    def __init__(self, k, theta):
        self.k = k
        self.heap = []
        self.set = set()
        self.theta = theta

    def add(self, sequence, quality, extend, error_delta):
        if sequence not in self.set:
            heapq.heappush(self.heap, (quality, sequence, extend, error_delta))
            self.set.add(sequence)

    def get_top_k_non_redundant(self, data, k, context=None):
        filtered_result = filter_results(
            self.heap, data, self.theta, k,
            context=context,
        )
        return heapq.nlargest(k, filtered_result)
