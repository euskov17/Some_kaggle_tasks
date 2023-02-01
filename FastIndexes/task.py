# coding: utf-8
"""
    TASK04* MCS, Bayes 101
    Это творческая бонусная задача.

    В базах данных и других областях наук о вычислительной технике,
    где для принятия тех или иных решений приходится иметь дело
    с упорядоченными массивами данных, знают разные способы определения
    адреса (т.е. позиции) элемента по тому самому значению, по которому
    данные упорядочены.

    Однако несколько лет назад предложили решать задачу определения
    позиции элемента в упорядоченном массиве по значению не с помощью,
    скажем, B-деревьев, а обучать вместо них специальную функцию.

    https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2018_2019/papers/Kraska_SIGMOD_2018.pdf

    Пример: у нас есть отсортированные большие целые числа (даты в логах),
    и нам нужно найти по заданному числу его позицию в массиве (т.е. чтобы
    быстро узнать, что написано в логах вокруг этой даты, например).

    Всем знакома эта задача -- это регрессия. Но только здесь нам нужно
    а) максимально точно предсказывать индексы (т.е. оверфиттить можно сколько угодно),
    б) использовать минимум параметров (то есть памяти),
    в) использовать быстрый метод (сложные пошаговые действия и привлечение
       специального оборудования вроде GPU -- замедляют работу).

    Вам предлагается решить эту задачу по-своему.

    TODO: ОБУЧИТЬ ИНДЕКС ТАК, ЧТОБЫ
    TODO: - MAE была меньше 20 на обоих наборах данных
    TODO: - уложиться примерно в 15 секунд (мой CPU: 1.80GHz; 16 секунд уже не ОК)
    TODO: - число параметров каждого индекса -- было не более 3000 вещественных чисел

    В качестве базы -- реализовано что-то вроде RMI. Разберитесь сами,
    что тут происходит. Можно посмотреть в статью, ссылка на которую выше --
    идея предельно простая -- несколько уровней дерева, в узлах модели,
    которые приближают -- каждая свой -- участок данных.

    Для начала рекомендую посмотреть и запустить `submodel.py` и с особым вниманием
    отнестись к тому, что именно на осях у графиков.

    Потом погонять этот файл `task.py`.

    Потом начать возиться с `index.py` (ну и `submodel.py` тоже, разумеется).

    ---

    Идеи:
    - может, можно подбирать для разных узлов разные модели?
    - может, можно не только делать .fit, но и подбирать гиперпараметры?
    - может, можно разбивать на участки для подгонки одной моделью как-то поумнее?

"""

"""
Что я сделал:

Я использую Ridge-регрессию: 
причем я уменьшаю постепенно с каждым шагом параметр alpha, чтобы у нас были 
более сглаженные кривые в начале, а потом все менее и менее регуляризованные.
Также я добавил еще 1 признак np.sqrt(num), чтобы лучше предсказывать.

И это дало свой результат, в частности это улучшение улучшило MAE на первом датасете на 4.
И по отдельности улучшения дают очень мало, а вместе как раз неплохой результат.

Дальше подбором гиперпараметров получилось добиться такого результата.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error

from index import LearnedIndex


def plot_actual_vs_predicted(indices_true: np.ndarray, indices_predicted: np.ndarray, filename: str):
    plt.plot(indices_true, color="black", linestyle="dashed", label="true")
    plt.plot(indices_predicted, color="blue", label="pred")
    plt.xlabel("Indices")
    plt.ylabel("Predicted Indices")
    plt.legend(loc="upper left")
    plt.title(f"Index fitted to '{filepath}' dataset")
    plt.savefig(filename)
    plt.clf()


def process_predictions(data: np.ndarray, model: LearnedIndex):
    indices_predicted = []

    for element in data:
        indices_predicted.append(model.predict(element))

    indices_predicted = np.array(indices_predicted)
    return np.floor(indices_predicted).astype(int)


if __name__ == "__main__":

    start_time = time.time()
    for filepath in ["set1_step10000.npz", "set2_step10000.npz"]:
        # зачитываем числа из нампаевского бинарника
        numbers = np.load(filepath)["numbers"]
        print(filepath, "Set length:", len(numbers))

        # просто индексы, которые в идеале должны быть предсказаны
        actual_true_indices = np.arange(0, len(numbers), dtype=int)

        # using Ridge Regression inside
        li = LearnedIndex(stages=5,  # сколько уровней в иерархии (в дереве подмоделей)
                          branching_factor=5)  # сколько "детей" у нелистов в дереве

        li.fit(numbers)

        start = time.time()
        actual_indices_predicted = process_predictions(numbers, li)
        end = time.time()

        exact_matches_count = np.count_nonzero(actual_indices_predicted == actual_true_indices)
        exact_matches_ratio = exact_matches_count / numbers.shape[0]

        print(f" Exact matches: {exact_matches_ratio * 100:.2f}%: {exact_matches_count}/{numbers.shape[0]}")
        print(" MAE:", mean_absolute_error(actual_true_indices, actual_indices_predicted))
        print(f" Elapsed: {end - start:.4f} seconds.")

        plot_actual_vs_predicted(actual_true_indices,
                                 actual_indices_predicted,
                                 Path(filepath).name + f"_st{li.stages}_bf{li.bf}.png")

        print("Total parameters:", li.params_total())

        print("--")
    print(f"Total time: {time.time() - start_time: .4f}")
