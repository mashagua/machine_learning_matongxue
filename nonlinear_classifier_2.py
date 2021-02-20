import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

rnd = np.random.RandomState(3)	# 为了演示，采用固定的随机
x_min, x_max = 0, 10

# 上帝函数 y=f(x)
def f(x):
	return x**5-22*x**4+161*x**3-403*x**2+36*x+938

def P(x):
	return f(x) + rnd.normal(scale=30,size=x.shape)
x=rnd.uniform(x_min,x_max,50)
y=P(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2021)
