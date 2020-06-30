import numpy as np
from matplotlib import pyplot as plt

d = 2

class Dataset:
    def __init__(self, N):
        self.X = 2*np.random.rand(N, d+1)-1
        self.X[:,0] = 1.0
        X = self.X
        xa = 2*np.random.rand() - 1
        ya = 2*np.random.rand() - 1
        xb = 2*np.random.rand() - 1
        yb = 2*np.random.rand() - 1
        xa, xb = min(xa, xb), max(xa, xb)
        self.xa = xa
        self.ya = ya
        self.xb = xb
        self.yb = yb
        self.a = (yb-ya)/(xb-xa)
        self.y = self.evaluate(X)
        
    def evaluate(self, X):
        return 2*(X[:,2] > self.ya + (X[:,1] - self.xa)*self.a)-1
        
    def plot(self, y=None):
        above = self.X[np.where(self.y < 0)]
        below = self.X[np.where(self.y >= 0)]
        plt.scatter(above[:,1], above[:,2], marker="x")
        plt.scatter(below[:,1], below[:,2], marker="o")
        if not y is None:
            wrong = np.where(y != self.y)
            plt.scatter(self.X[wrong,1], self.X[wrong,2], c="red")
        plt.plot([self.xa, self.xb], [self.ya, self.yb])
        plt.show()
        
        
def fit_ls(X, y):
    return np.linalg.solve(X.T@X, X.T@y)

def evaluate(X, w):
    return np.sign(X@w)

N = 100
gs = np.zeros([1000, N])

total_Ein = 0.0
for i in range(1000):
    ds = Dataset(N)
    w_ls = fit_ls(ds.X, ds.y)
    total_Ein += np.sum(evaluate(ds.X, w_ls) != ds.y)/N
    
print(total_Ein/1000)