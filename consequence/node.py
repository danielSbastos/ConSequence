import random

from consequence.utils import find_LCS, sequence_mutable_to_immutable, compute_quality, compute_sequence_expand

import consequence.conf as conf


class Node():
    def __init__(self, intent, parent, node_hashmap):
        self.intent = intent
        self.node_hashmap = node_hashmap
        self.depth = 0 if parent is None else parent.depth + 1

        self._quality = None
        self._error_delta = None
        self._extend = None
        self._size_class_0 = None
        self._size_class_1 = None
        self._candidate_sequences_expand = None

        if parent != None:
            self.parents = [parent]
            parent.children.append(self)
        else:
            self.parents = []

        self.children = []
        self.number_visits = 1
        self.dead_end = False

    def _ensure_quality_computed(self):
        if self._quality is None:
            q, ed, e, n0, n1 = self.get_extend_and_quality(self.intent)
            self._quality, self._error_delta, self._extend = q, ed, e
            self._size_class_0, self._size_class_1 = n0, n1

    @property
    def quality(self):
        self._ensure_quality_computed()
        return self._quality

    @property
    def error_delta(self):
        self._ensure_quality_computed()
        return self._error_delta

    @property
    def extend(self):
        self._ensure_quality_computed()
        return self._extend

    @property
    def size_class_0(self):
        self._ensure_quality_computed()
        return self._size_class_0

    @property
    def size_class_1(self):
        self._ensure_quality_computed()
        return self._size_class_1

    @property
    def candidate_sequences_expand(self):
        if self._candidate_sequences_expand is None:
            self._initialize_candidates()
        return self._candidate_sequences_expand

    def _initialize_candidates(self):
        if self.intent is not None:
            candidate_sequences_expand = compute_sequence_expand(tuple(self.intent), tuple(self.extend))
        else:
            candidate_sequences_expand = compute_sequence_expand(self.intent, tuple(self.extend))

        self._candidate_sequences_expand = [seq for _, seq in candidate_sequences_expand]

    def get_normalized_quality(self):
        return self.quality

    def get_extend_and_quality(self, subsequence):
        if self.intent is None:
            return 0, -1, [], 0, 0
        return compute_quality(sequence_mutable_to_immutable(subsequence))

    def is_fully_expanded(self):
        return len(self.candidate_sequences_expand) == 0

    def is_terminal(self):
        return len(self.extend) == len(conf.model.data)

    def is_dead_end(self):
        if self.is_terminal() or self.dead_end:
            self.dead_end = True
            return True

        if not self.is_fully_expanded():
            return False

        for child in self.children:
            if not child.is_dead_end():
                return False

        self.dead_end = True
        return True

    def expand(self):
        if self._candidate_sequences_expand is None:
            self._initialize_candidates()

        if len(self._candidate_sequences_expand) == 0:
            sequence_children = tuple()
        else:
            random_object_idx = random.randint(0, len(self._candidate_sequences_expand) - 1)
            random_object = self._candidate_sequences_expand.pop(random_object_idx)

            if self.intent == None:
                sequence_children = sequence_mutable_to_immutable(random_object)
            else:
                sequence_children = sequence_mutable_to_immutable(find_LCS(random_object, self.intent))

            if len(sequence_children) == 0:
                sequence_children = tuple()

        if sequence_children in self.node_hashmap:
            child = self.node_hashmap[sequence_children]
            child.parents.append(self)
            self.children.append(child)
        else:
            child = Node(sequence_children, self, self.node_hashmap)
            self.node_hashmap[sequence_children] = child

        return child

    def update(self, reward):
        current_quality = self.quality
        self._quality = (self.number_visits * current_quality + reward) / (
                self.number_visits + 1)
        self.number_visits += 1