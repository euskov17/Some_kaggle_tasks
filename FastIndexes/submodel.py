# coding: utf-8

import numpy as np
from frozendict import frozendict
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


class SubModel(object):
    """ Это "атомарная модель", которая будет приближать набор данных на своём "участке". """

    def __init__(self, model_type=LinearRegression, model_params=frozendict({"fit_intercept": True, "n_jobs": -1})):
        """
            BayesianRidge?
            SGDRegression (будет ли принципиально отличаться от LinearRegression? там есть разные loss)?
            SVR (какие у неё параметры и сколько их?
                 как долго она работает на куске данных такого размера)? внимание к kernel, gamma, epsilon
            IsotonicRegression (сколько у неё параметров? как долго она будет работать на нашем примере?)?
            другие варианты?
        """
        self.model = model_type(**model_params)
        self.span_size = None  # тоже параметр, кстати

    def fit(self, ordered_subset: np.ndarray):
        """
            Saves the span size and trains the submodel to predict normalized indices.

            Parameters
            ----------
            ordered_subset : one-dimensional numpy array of shape (span_size,)
                Training data.
        """
        self.span_size = ordered_subset.shape[0]
        y = np.arange(0, self.span_size) / self.span_size
        self.model.fit(ordered_subset, y)

    def predict(self, values: np.ndarray) -> np.ndarray:
        # predicting
        raw_predictions = self.model.predict(values)
        # fixing values that are out of bounds
        temporary_to_indices = np.floor(raw_predictions * self.span_size)
        predictions = np.minimum(np.maximum(temporary_to_indices, 0), self.span_size - 1) / self.span_size
        return predictions

    def params_total(self):
        # todo: note that this code is submodel-specific
        try:
            return len(self.model.coef_) + (1 if self.model.fit_intercept else 0) + 1  # the last for span size
        except AttributeError as ae:
            return 0

    def __repr__(self):
        return f"SubModel(model={self.model}, span_size={self.span_size}, params_count={self.params_total()})"


if __name__ == "__main__":
    from sklearn.metrics import mean_absolute_error
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import itertools

    print(" SINGLE MODEL PREDICTION -- testing")

    # синтетические данные со скачками
    numbers = np.log2(np.arange(1000) ** 2 + 5) + np.array(([0] * 500) + ([5] * 500))

    plt.plot(numbers, color="green", label="data")
    plt.legend(loc="upper left")
    plt.title("Data sequence")
    plt.xlabel("Indices")
    plt.ylabel("Values")
    plt.show()

    # приближаем одной моделью -- очень старательно оверфиттим
    param_grid = {
        "gamma": [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0],
        "C": [0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0],
        "epsilon": [0.1, 0.08, 0.05, 0.01, 0.005]
    }

    # задаём фиксированный порядок имён параметров
    keys = list(param_grid.keys())
    lists = tuple([param_grid[k] for k in keys])

    best_matches, best_matches_params, best_actual_indices_predicted, best_mae = -1, None, None, None
    actual_true_indices = np.arange(0, len(numbers), dtype=int)

    for tuple_vals in tqdm(itertools.product(*lists)):

        hyperparams = {k: v for k, v in zip(keys, tuple_vals)}
        hyperparams["kernel"] = "rbf"

        # создаём
        sm = SubModel(SVR, hyperparams)

        # обучаем
        sm.fit(numbers)

        # предсказываем вещественные оценки
        indices_predicted = sm.predict(numbers)

        # округляем для индексов
        actual_indices_predicted = np.floor(indices_predicted * len(numbers)).astype(int)

        # оцениваем качество
        matches = np.count_nonzero(actual_indices_predicted == actual_true_indices) / numbers.shape[0]
        mae = mean_absolute_error(actual_true_indices, actual_indices_predicted)

        # "сохраним в университетах лучших"
        if matches > best_matches:
            best_matches, best_matches_params, best_mae = matches, hyperparams, mae
            best_actual_indices_predicted = actual_indices_predicted

    print(" Exact matches:", best_matches * 100, "%")
    print(" MAE:", best_mae)
    print(" Best params:", best_matches_params)

    plt.plot(actual_true_indices, color="black", linestyle="dashed", label="true")
    plt.plot(best_actual_indices_predicted, color="blue", label="pred")
    plt.xlabel("Indices")
    plt.ylabel("Predicted indices")
    plt.legend(loc="upper left")
    plt.title("Predicted indices: single model fitted to the whole dataset")
    plt.show()
