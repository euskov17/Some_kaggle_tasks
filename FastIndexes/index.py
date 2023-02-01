# coding: utf-8

import pickle
import numpy as np
from sklearn.linear_model import Ridge
from submodel import SubModel


class LearnedIndex(object):
    """
        The simple analogue of the Recursive Model Index
        https://arxiv.org/pdf/1712.01208.pdf
    """

    def __init__(self, stages: int = 2, branching_factor: int = 2):

        self.bf, self.stages = branching_factor, stages
        model_type = Ridge

        def gen_tree(stage: int) -> dict:
            """ Recursively building a submodels tree """

            if stage == stages:
                return None

            # creating the current node
            result = {"stage": stage,  # уровень в дереве, отсчитывается с нуля с корня
                      "model": SubModel(model_type,
                                        {"fit_intercept": True,
                                         "alpha": 10 ** (- 2 - stage * 2)}),  # подмодель в данном узле
                      "left_right": None,  # границы участка, за который отвечает подмодель
                      "submodels": []}  # подмодели ниже уровнем (если есть)

            # creating the child nodes
            for part in range(branching_factor):
                next_stage_tree = gen_tree(stage + 1)
                if next_stage_tree is not None:
                    result["submodels"].append(next_stage_tree)

            return result

        # initializing the whole tree structure
        self.submodels: dict = gen_tree(0)

    def fit(self, data: np.ndarray):
        """
            Trains all the submodels to predict unnormalized indices.

            Parameters
            ----------
            data : one-dimensional numpy array of shape (span_size,)
                Training data.
        """

        # using the queue to traverse the submodels and their indices spans
        queue = [(self.submodels, (0, len(data)))]
        while len(queue) > 0:

            current, index_range = queue.pop(0)
            interval = np.array(data[index_range[0]: index_range[1]])
            data_chunk = np.vstack([interval, interval ** .5]).T
            current["left_right"] = [index_range[0], index_range[1]]
            current["model"].fit(data_chunk)

            if current["stage"] != self.stages - 1:

                predicted_indices = current["model"].predict(data_chunk)
                split_values = np.arange(1, self.bf, step=1) / self.bf
                split_ids = [0] + list(np.sum(predicted_indices < split_values[:, None], axis=1)) + [len(data_chunk)]

                for child_id, submodel in enumerate(current["submodels"]):

                    if split_ids[child_id + 1] > split_ids[child_id]:
                        queue.append((submodel, (index_range[0] + split_ids[child_id],
                                                 index_range[0] + split_ids[child_id + 1])))

    def predict(self, single_value):

        current = self.submodels
        scaled_value = None
        for stage in range(self.stages):
            param = np.array([single_value, single_value ** .5]).reshape(-1, 1).T
            value = current["model"].predict(param)[0]

            if stage < self.stages - 1:

                selected_submodel = None

                for i, threshold in enumerate(np.arange(1, self.bf + 1) / self.bf):
                    if value < threshold:
                        selected_submodel = i
                        break
                current = current["submodels"][selected_submodel]
            else:
                # scaling
                val_min, val_max = current["left_right"][0], current["left_right"][1]
                scaled_value = value * (val_max - val_min) + val_min

        return scaled_value

    def _total_size(self):
        size_estimate = len(pickle.dumps(self.submodels))
        return size_estimate

    def params_total(self):

        unprocessed_queue = [self.submodels]
        params_count_sum = 0

        while len(unprocessed_queue) > 0:
            subtree = unprocessed_queue.pop(0)
            params_count_sum += subtree["model"].params_total()
            unprocessed_queue.extend(subtree["submodels"])

        return params_count_sum
