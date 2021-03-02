# -*- coding: utf-8 -*-
# @Time : 2021/3/2 8:33
# @Author : mashagua
# @Email : Shaoguang.MA@cn.bosch.com
import numpy as np
from sklearn.datasets import load_iris
from collections import Counter


def calc_gini(leaf):
    types = leaf[:, 2]
    c = Counter(types)
    p1 = c[0] / types.size
    p2 = c[1] / types.size
    p3 = c[2] / types.size
    return 1 - (p1**2 + p2**2 + p3**2)


iris = load_iris()
x = iris.data[:, [1, 2]]
y = iris.target
flowers = np.c_[x, y]
