import pytest
import numpy as np
import random
from scipy.stats import qmc
import math

def test_diff_evolution_part_2():
    from diff_evolution import differential_evolution
    SEED = 21
    random.seed(SEED)
    np.random.seed(SEED)

    def rastrigin(array, A=10):
        return A * 2 + (array[0] ** 2 - A * np.cos(2 * np.pi * array[0])) + (array[1] ** 2 - A * np.cos(2 * np.pi * array[1]))
  
    def griewank(array):
        term_1 = (array[0] ** 2 + array[1] ** 2) / 2
        term_2 = np.cos(array[0]/ np.sqrt(2)) * np.cos(array[1]/ np.sqrt(2))
        return 1 + term_1 - term_2
  
    def rosenbrock(array):
        return (1 - array[0]) ** 2 + 100 * (array[1] - array[0] ** 2) ** 2

    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] == 9.950103123657073e-07
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='best1', selection_setting='random_selection'))[-1][1] == 0.0
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] == 1.0365393876554663e-08
    assert list(differential_evolution(griewank, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] == 8.105971449623439e-11
    assert list(differential_evolution(griewank, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='best1', selection_setting='random_selection'))[-1][1] == 0.0
    assert list(differential_evolution(griewank, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] == 1.0701439734361884e-12
    assert list(differential_evolution(rosenbrock, np.array([[0, 2], [0, 2]]), init_setting='random', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] == 8.11148340492266e-05
    assert list(differential_evolution(rosenbrock, np.array([[0, 2], [0, 2]]), init_setting='random', mutation_setting='best1', selection_setting='random_selection'))[-1][1] == 1.6272749339911577e-10
    assert list(differential_evolution(rosenbrock, np.array([[0, 2], [0, 2]]), init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] == 2.8810880486788108e-05
