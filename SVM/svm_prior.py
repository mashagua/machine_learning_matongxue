# -*- coding: utf-8 -*-
# @Time : 2021/3/1 15:12
# @Author : mashagua
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib.colors import ListedColormap
# x = np.array([[-1.8, 0.6], [0.48, -1.36], [3.68, -3.64],
#               [1.44, 0.52], [3.42, 3.5], [-4.18, 1.68]])
# z = x**2
# y = np.array([1, 1, -1, 1, -1, -1])
x = np.array([[-1, -1], [-1, 2],[2, 2], [2, -1]])
z = np.hstack((x**2, (x[:, 0] * x[:, 1]).reshape(-1, 1)))
y = np.array([1, -1, 1, -1])

m = z.shape[0]
lambdas = cp.Variable(m)
constraint_1 = [lambdas[i] >= 0 for i in range(m)]
constraint_2 = [lambdas @ y.T == 0]
yZ = y.reshape(-1, 1) * z
K = yZ @ yZ.T
objective_dual = cp.Minimize(
    1 /
    2 *
    cp.quad_form(
        lambdas,
        K) -
    cp.sum(lambdas))
prob_dual = cp.Problem(objective_dual, constraint_1 + constraint_2)
prob_dual.solve()
lambdas = lambdas.value
support_y = y[lambdas > 0.0001]
support_z = z[lambdas > 0.0001]
support_lambdas = lambdas[lambdas > 0.0001]
w = np.sum((support_lambdas * support_y).reshape(-1, 1) * support_z, axis=0)
b = support_y[0] - np.sum((support_lambdas * support_y).reshape(-1, 1)
                          * support_z @ support_z[0].reshape(-1, 1))
print('lambda1:{:3f}'.format(lambdas[0]))
