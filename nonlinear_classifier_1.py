import matplotlib.pyplot as plt
import numpy as np
rnd = np.random.RandomState(2021)
x_min, x_max = -1, 1


# 上帝函数
def f(x):
    return x**2

def P(X):
    return f(X)+rnd.normal(scale=0.1, size=X.shape)

X = rnd.uniform(x_min,x_max,10)
y = P(X)
plt.rcParams.update({'font.size':14})
fig,ax=plt.subplots(figsize=(6,3))
plt.subplots_adjust(left=0.25, right=0.75, top=0.999, bottom=0.08)
ax.set(xticks=[], yticks=[])
ax.set_xlabel('$x$'),ax.set_ylabel('$y$')
ax.set_xlim(x_min, x_max)

# 绘制数据集
ax.scatter(x=X, y=y)

# 绘制上帝函数
xx = np.linspace(x_min, x_max)
ax.plot(xx, f(xx), 'k--')

plt.show()
