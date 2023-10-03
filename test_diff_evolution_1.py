import pytest
import numpy as np
import random
from scipy.stats import qmc
import math

def test_diff_evolution_part_1():
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

    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] == 1.290061391046038e-10
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] == 2.2518667819326765e-08
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Halton', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] == 5.195389007894846e-09
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Sobol', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] == 8.091793901598976e-09
    assert list(differential_evolution(griewank, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] == 6.583622536027178e-14
    assert list(differential_evolution(griewank, np.array([[-20, 20], [-20, 20]]), init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] == 1.3922196728799463e-13
    assert list(differential_evolution(griewank, np.array([[-20, 20], [-20, 20]]), init_setting='Halton', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] == 1.249000902703301e-13
    assert list(differential_evolution(griewank, np.array([[-20, 20], [-20, 20]]), init_setting='Sobol', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] == 1.354472090042691e-14
    assert list(differential_evolution(rosenbrock, np.array([[0, 2], [0, 2]]), init_setting='random', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] == 3.472211458436623e-06
    assert list(differential_evolution(rosenbrock, np.array([[0, 2], [0, 2]]), init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] == 0.00022435084404149274
    assert list(differential_evolution(rosenbrock, np.array([[0, 2], [0, 2]]), init_setting='Halton', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] == 2.4486440221569863e-05
    assert list(differential_evolution(rosenbrock, np.array([[0, 2], [0, 2]]), init_setting='Sobol', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] == 1.989204789662423e-06
