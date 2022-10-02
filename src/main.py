import numpy as np
from scipy import stats as st
import math
import random as rd
import matplotlib.pyplot as plt
# q = np.array([2, 1, 6, 1, 4, 6, 2, 2, 3, 2, 3, 2, 5, 2, 2, 1, 4, 6, 1, 6, 5, 4, 6, 6, 6, 2, 2, 3,
# 3, 6, 1, 4, 1, 5, 1, 2, 5, 5, 2, 2, 6, 2, 5, 2, 4, 2, 4, 5, 3, 5, 6, 1, 5, 4, 3, 2,
# 2, 4, 4, 3, 5, 2, 5, 2, 1, 1, 5, 1, 3, 2, 3, 3, 2, 3, 4, 4, 5, 3, 3, 1, 2, 1, 3, 5,
# 2, 5, 4, 2, 4, 3, 1, 5, 5, 1, 4, 4, 6, 2, 6, 3, 5, 3, 3, 3, 6, 1, 1, 2, 6, 1, 3, 3,
# 4, 3, 1, 5, 3, 2, 1, 3])

# print(st.shapiro(q))
# n = 120
# p = 1/6
# q = 5/6

# x = [ 20,27,24,16,19,14]

# for i in x:
#     z = (i - n*p )/ math.sqrt(n*p*q)
#     print(z)


# x = np.array([rd.randint(0, 100) for i in range(60)])
# y = np.array([rd.randint(0, 100) for i in range(60)])

x=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y=np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
a = (np.dot(x,y) - (y.sum()*x.sum()) / len(x) )/ ((x**2).sum()-x.sum()**2 / len(x))
b = (y.sum() - a* x.sum()) / len(x)

print(np.var(x,ddof=0))
print(np.var(y,ddof=0))
print(np.std(x,ddof=0))
print(np.std(y,ddof=0))
print(np.cov(x,y,ddof=0))
print(np.cov(x,y,ddof=0)[0][1]/(np.std(x,ddof=0)*np.std(y,ddof=0)))

print("\ncov = {}".format(np.cov(x,y,ddof=0)[0][1]/(np.std(x,ddof=0)*np.std(y,ddof=0))))
print("y = {}x + {}".format(a,b))
print("決定係数 = {}".format(np.cov(x,y,ddof=0)[0][1]/(np.std(x,ddof=0)*np.std(y,ddof=0))*np.cov(x,y,ddof=0)[0][1]/(np.std(x,ddof=0)*np.std(y,ddof=0))))
plt.figure(figsize=(16,9))
plt.scatter(x,y,color="k")
plt.plot([0,x.max()],[b,a*x.max()+ b])
plt.savefig('../img/stats_saisyonijouhou.png')
plt.show()