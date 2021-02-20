import numpy as np
#set the parameters
xk=np.array([10])
eta=0.1
epochs=5000
epsilon=0.0000001

def f(x):
    return x[0]*x[0]

def df(x):
    return 2*x

for i in range(epochs):
    dfxk=df(xk)
    if np.linalg.norm(dfxk)<epsilon:
        print('经过 {} 次迭代，梯度下降法运行完毕'.format(i + 1))
        print('结果为 xk = {} ，f(xk) = {}'.format(xk, f(xk)))
        break
    xk = xk - eta * dfxk