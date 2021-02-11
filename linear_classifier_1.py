import numpy as np
w, w0 = np.array([0, 0, 0, 0, 0]), 0
def d(x):
    return np.dot(w, x)+w0

def sign(x):
    return 1 if x >= 0 else -1

def h(x):
    return sign(d(x))

def clf_score(x,y):
    score = 0
    for xi, yi in zip(x, y):
        score += yi*h(xi)
    return score


x = np.array([[-1.8, 0.6], [0.48, -1.36], [3.68, -3.64], [1.44, 0.52], [3.42, 3.5], [-4.18, 1.68], ])
Z = np.hstack((x ** 2, (x[:, 0] * x[:, 1]).reshape(-1, 1), x,))  # 转换为 z1z2z3z4z5 坐标系中的点
y = np.array([1, 1, -1, 1, -1, -1, ])
print('转换后的数据集为：')
zy=np.hstack((Z,y.reshape(-1,1)))
best_w, best_w0 = w, w0
best_cs = clf_score(Z, y)
epochs = 100
for _ in range(epochs):
    for zi,yi in zip(Z,y):
        if yi*d(zi)<=0:
            w,w0=w+yi*zi,w0+yi
            cs=clf_score(Z,y)
            if cs>best_cs:
                best_cs=cs
                best_w,best_w0=w,w0
            break

w,w0=best_w,best_w0
print("感知机口袋算法结果为：")
print('w0 = {:.02f}, w1 = {:.02f}, w_2 ={:.02f}, w3={:.02f}, w4={:.02f}, w5={:.02f}'.format(w0, w[0], w[1], w[2], w[3], w[4]))
