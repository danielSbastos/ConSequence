import functools

import numpy as np

import consequence.conf as conf


def sequence_mutable_to_immutable(sequence):
    return tuple([frozenset(i) for i in sequence])


def sequence_immutable_to_mutable(sequence):
    return [set(i) for i in sequence]


def is_subsequence_contiguous(a, b):
    if len(a) > len(b):
        return False

    if len(a) == 0:
        return True

    for start_idx in range(len(b) - len(a) + 1):
        match = True
        for i in range(len(a)):
            if not a[i].issubset(b[start_idx + i]):
                match = False
                break
        if match:
            return True

    return False


def is_subsequence_windowed(a, b, max_gap=None):
    if len(a) > len(b):
        return False

    if len(a) == 0:
        return True

    prev_positions = [set()]

    for j in range(len(b)):
        if a[0].issubset(b[j]):
            prev_positions[0].add(j)

    if not prev_positions[0]:
        return False

    for i in range(1, len(a)):
        prev_positions.append(set())

        for prev_j in prev_positions[i-1]:
            for j in range(prev_j + 1, min(prev_j + max_gap + 2, len(b))):
                if a[i].issubset(b[j]):
                    prev_positions[i].add(j)

        if not prev_positions[i]:
            return False

    return True


def is_subsequence_non_contiguous(a, b):
    if len(a) > len(b):
        return False

    if len(a) == 0:
        return True

    i_a, i_b = 0, 0

    while i_a < len(a) and i_b < len(b):
        if a[i_a].issubset(b[i_b]):
            i_a += 1
        i_b += 1

    return i_a == len(a)


def is_subsequence(a, b, max_gap=None):
    if max_gap == 0:
        return is_subsequence_contiguous(a, b)
    elif max_gap == -1:
        return is_subsequence_non_contiguous(a, b)
    else:
        return is_subsequence_windowed(a, b, max_gap)


def encode_data(data, item_to_encoding):
    missing_items = set()
    missing_items_count = 0

    for line in data:
        itemsets_to_remove = []
        for i in range(1, len(line)):
            itemset = line[i]
            if len(itemset) == 0:
                itemsets_to_remove.append(i)
                continue

            encoded_itemset = set()
            for item in itemset:
                if item in item_to_encoding:
                    encoded_itemset.add(item_to_encoding[item])
                else:
                    missing_items.add(item)
                    missing_items_count += 1

            if len(encoded_itemset) > 0:
                line[i] = encoded_itemset
            else:
                itemsets_to_remove.append(i)

        for i in sorted(itemsets_to_remove, reverse=True):
            if i < len(line):
                del line[i]

    if missing_items:
        print(f"Warning: {missing_items_count} items not found in vocabulary (skipped): {sorted(list(missing_items))[:10]}{'...' if len(missing_items) > 10 else ''}")

    return data


def decode_sequence(sequence, encoding_to_item):
    return_sequence = []

    for itemset in sequence:
        decoded_itemset = set()
        for item in itemset:
            decoded_itemset.add(encoding_to_item[item])
        return_sequence.append(decoded_itemset)
    return return_sequence


def decode_sequences(results, encoding_to_item):
    return_results = []
    for result in results:
        return_results.append((result[0], decode_sequence(result[1], encoding_to_item)))
    return return_results


def encode_items(items):
    item_to_encoding = {}
    encoding_to_item = {}
    new_items = set()

    for i, item in enumerate(items):
        item_to_encoding[item] = i
        encoding_to_item[i] = item
        new_items.add(i)

    return new_items, item_to_encoding, encoding_to_item


def extract_items(data):
    items = set()
    for sequence in data:
        for itemset in sequence[1:]:
            for item in itemset:
                items.add(item)
    return sorted(list(items))


def print_results(results):
    sum_result = 0
    for result in results:
        pattern_display = ''
        for itemset in result[1]:
            pattern_display += repr(set(itemset))

        sum_result += result[0]

        print('Quality: {}, Extent: {}, Pattern_Delta: {}, Pattern: {}'.format(result[0], result[2], result[3], pattern_display))

    print('Average score :{}'.format(sum_result / len(results)))


def print_results_decode(results, encoding_to_items):
    decoded_results = []
    for result in results:
        decoded_result = []
        decoded_result.append(result[0])
        decoded_result.append(decode_sequence(result[1], encoding_to_items))
        decoded_result.append(len(result[2]))
        decoded_result.append(result[3])
        decoded_results.append(decoded_result)

    print_results(decoded_results)


def get_quality(support, data, extend, target_class=None):
    if target_class is None:
        target_class = conf.model.target_class

    soft_errors = conf.model.soft_errors
    y_true = target_class[:, 0]
    extend_arr = np.array(extend, dtype=int)

    subgroup_class_0 = soft_errors[extend_arr][y_true[extend_arr] == 0]
    subgroup_class_1 = soft_errors[extend_arr][y_true[extend_arr] == 1]

    subgroup_size_class_0 = len(subgroup_class_0)
    subgroup_size_class_1 = len(subgroup_class_1)
    baseline_class_0 = soft_errors[y_true == 0]
    baseline_class_1 = soft_errors[y_true == 1]

    if (subgroup_size_class_0 == 0 or subgroup_size_class_1 == 0 or (subgroup_size_class_0 == len(baseline_class_0) and
        subgroup_size_class_1 == len(baseline_class_1))):
        return 0.0, 0.0, subgroup_size_class_0, subgroup_size_class_1

    error_delta = abs(subgroup_class_0.mean() - subgroup_class_1.mean())
    std_0 = subgroup_class_0.std()
    std_1 = subgroup_class_1.std()
    s_g = error_delta / max(std_0, std_1)

    mean_diff_g0_b = abs(subgroup_class_0.mean() - baseline_class_0.mean())
    d_0 = mean_diff_g0_b / baseline_class_0.std()
    mean_diff_g1_b = abs(subgroup_class_1.mean() - baseline_class_1.mean())
    d_1 = mean_diff_g1_b / baseline_class_1.std()
    d_g = max(d_0, d_1)

    total_subgroup = subgroup_size_class_0 + subgroup_size_class_1
    p0 = subgroup_size_class_0 / total_subgroup
    p1 = subgroup_size_class_1 / total_subgroup
    class_balance_score = (4 * p0 * p1)

    support_penalty = (total_subgroup / len(soft_errors)) ** conf.conf.support_penalty

    quality = s_g * d_g * class_balance_score * support_penalty
    sigmoid_quality = 1 / (1 + np.e ** (-(quality - 2)))

    return sigmoid_quality, error_delta, subgroup_size_class_0, subgroup_size_class_1


def compute_sg_dg_statistic(extend, target_class, soft_errors):
    soft_errors = np.asarray(soft_errors)
    y_true = target_class[:, 0]
    extend_arr = np.array(extend, dtype=int)

    subgroup_class_0 = soft_errors[extend_arr][y_true[extend_arr] == 0]
    subgroup_class_1 = soft_errors[extend_arr][y_true[extend_arr] == 1]

    subgroup_size_class_0 = len(subgroup_class_0)
    subgroup_size_class_1 = len(subgroup_class_1)
    baseline_class_0 = soft_errors[y_true == 0]
    baseline_class_1 = soft_errors[y_true == 1]

    min_per_class = 15
    if (subgroup_size_class_0 < min_per_class or
        subgroup_size_class_1 < min_per_class or
        (subgroup_size_class_0 == len(baseline_class_0) and
         subgroup_size_class_1 == len(baseline_class_1))):
        return 0.0

    std_0 = subgroup_class_0.std()
    std_1 = subgroup_class_1.std()
    if std_0 <= 0 or std_1 <= 0:
        return 0.0
    baseline_std_0 = baseline_class_0.std()
    baseline_std_1 = baseline_class_1.std()
    if baseline_std_0 <= 0 or baseline_std_1 <= 0:
        return 0.0

    mean_diff_g = abs(subgroup_class_0.mean() - subgroup_class_1.mean())
    s_g = mean_diff_g / max(std_0, std_1)

    mean_diff_g0_b = abs(subgroup_class_0.mean() - baseline_class_0.mean())
    d_0 = mean_diff_g0_b / baseline_std_0
    mean_diff_g1_b = abs(subgroup_class_1.mean() - baseline_class_1.mean())
    d_1 = mean_diff_g1_b / baseline_std_1
    d_g = max(d_0, d_1)

    return float(s_g * d_g)


@functools.lru_cache(maxsize=10000)
def compute_quality(subsequence):
    data = conf.model.data
    max_gap = conf.conf.max_gap

    support = 0
    extend = []

    for i, sequence in enumerate(data):
        if is_subsequence(subsequence, sequence, max_gap=max_gap):
            support += 1
            extend.append(i)

    quality, error_delta, size_class_0, size_class_1 = get_quality(support, data, extend)
    return quality, error_delta, extend, size_class_0, size_class_1


@functools.lru_cache(maxsize=1000)
def compute_sequence_expand(intent, extend):
    data = conf.model.data
    if intent is None:
        return tuple([[i, seq] for i, seq in enumerate(data) if i not in extend])
    max_gap = conf.conf.max_gap
    return tuple([[i, seq] for i, seq in enumerate(data) if i not in extend and not is_subsequence(intent, seq, max_gap=max_gap)])


def backtrack_LCS(C, seq1, seq2, i, j, lcs):
    if i == 0 or j == 0:
        return

    inter = seq1[i - 1].intersection(seq2[j - 1])

    if inter != set():
        if C[i - 1][j] == C[i][j]:
            return backtrack_LCS(C, seq1, seq2, i - 1, j, lcs)
        if C[i][j - 1] == C[i][j]:
            return backtrack_LCS(C, seq1, seq2, i, j - 1, lcs)
        else:
            lcs.insert(0, inter)
            return backtrack_LCS(C, seq1, seq2, i - 1, j - 1, lcs)

    if C[i][j - 1] > C[i - 1][j]:
        return backtrack_LCS(C, seq1, seq2, i, j - 1, lcs)
    else:
        return backtrack_LCS(C, seq1, seq2, i - 1, j, lcs)


def find_LCS_contiguous(seq1, seq2):
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return []

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    end_i = 0
    end_j = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            inter = seq1[i - 1].intersection(seq2[j - 1])
            if inter:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_i = i
                    end_j = j
            else:
                dp[i][j] = 0

    if max_len == 0:
        return []

    lcs = []
    i, j = end_i, end_j
    while i > 0 and j > 0 and dp[i][j] > 0:
        inter = seq1[i - 1].intersection(seq2[j - 1])
        if not inter:
            break
        lcs.append(inter)
        i -= 1
        j -= 1

    lcs.reverse()
    return lcs


def find_LCS_windowed(seq1, seq2, max_gap):
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return []

    dp = [[None] * (n + 1) for _ in range(m + 1)]

    best_len = 0
    best_end = (0, 0)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            inter = seq1[i - 1].intersection(seq2[j - 1])
            if not inter:
                continue

            best_prev = (1, 0, 0)

            for pi in range(max(0, i - max_gap - 1), i):
                for pj in range(max(0, j - max_gap - 1), j):
                    if pi == i and pj == j:
                        continue
                    if dp[pi][pj] is not None:
                        prev_len, _, _ = dp[pi][pj]
                        gap1 = (i - 1) - pi
                        gap2 = (j - 1) - pj
                        if gap1 <= max_gap and gap2 <= max_gap:
                            if prev_len + 1 > best_prev[0]:
                                best_prev = (prev_len + 1, pi, pj)

            dp[i][j] = best_prev

            if best_prev[0] > best_len:
                best_len = best_prev[0]
                best_end = (i, j)

    if best_len == 0:
        return []

    lcs = []
    i, j = best_end
    while i > 0 and j > 0 and dp[i][j] is not None:
        inter = seq1[i - 1].intersection(seq2[j - 1])
        if inter:
            lcs.append(inter)
        _, pi, pj = dp[i][j]
        if pi == 0 and pj == 0:
            break
        i, j = pi, pj

    lcs.reverse()
    return lcs


def find_LCS_non_contiguous(seq1, seq2):
    C = [[0 for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]

    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            inter = seq1[i - 1].intersection(seq2[j - 1])

            C[i][j] = max([C[i - 1][j - 1] + len(inter),
                           C[i - 1][j],
                           C[i][j - 1]])

    lcs = []
    backtrack_LCS(C, seq1, seq2, len(seq1), len(seq2), lcs)
    return lcs


def find_LCS(seq1, seq2, max_gap=None):
    if max_gap is None:
        max_gap = conf.conf.max_gap

    if max_gap == 0:
        return find_LCS_contiguous(seq1, seq2)
    elif max_gap == -1:
        return find_LCS_non_contiguous(seq1, seq2)
    else:
        return find_LCS_windowed(seq1, seq2, max_gap)


def filter_empty_sequences(data):
    return tuple([sequence_mutable_to_immutable(i[1:]) for i in data])
