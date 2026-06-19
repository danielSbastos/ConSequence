TOP_K = 5
TIME_BUDGET = 2 ** 30
ITERATIONS_LIMIT = 2 ** 30
MAX_LENGTH = 2 ** 30
THETA = 0.1
MIN_SUPPORT = 60
MAX_GAP = 2
SUPPORT_PENALTY = 0.0


class Conf:
    def __init__(
        self,
        top_k=None,
        time_budget=None,
        theta=None,
        iterations_limit=None,
        max_length=None,
        max_gap=None,
        support_penalty=None,
        min_support=None,
    ):
        self.top_k = TOP_K if top_k is None else top_k
        self.time_budget = TIME_BUDGET if time_budget is None else time_budget
        self.theta = THETA if theta is None else theta
        self.iterations_limit = ITERATIONS_LIMIT if iterations_limit is None else iterations_limit
        self.max_length = MAX_LENGTH if max_length is None else max_length
        self.max_gap = MAX_GAP if max_gap is None else max_gap
        self.support_penalty = SUPPORT_PENALTY if support_penalty is None else support_penalty
        self.min_support = MIN_SUPPORT if min_support is None else min_support


class Model:
    def __init__(self):
        self.labels = []
        self.data = None
        self.validation_data = None
        self.validation_target_class = None
        self.target_class = None
        self.global_mean_error = None
        self.global_std_error = None
        self.global_errors = None
        self.positive_class = None
        self.hard_errors = None
        self.soft_errors = None
        self.global_hard_error = None


conf = Conf()
model = Model()
