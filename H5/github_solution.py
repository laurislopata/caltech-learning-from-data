# %%
import numpy as np
import matplotlib.pyplot as plt

def random_point():
    return 2*np.random.rand()-1, 2*np.random.rand()-1

def random_line():
    x0, y0 = random_point()
    x1, y1 = random_point()
    a = (y1 - y0)/(x1 - x0)
    b = y1 - a*x1
    return b, a

def generate_dataset(N, b, a):
    X = 2*np.random.rand(N, 3)-1
    X[:,0] = 1
    y = np.sign(b + a*X[:,1] - X[:,2])
    return X, y

b, a = random_line()
X, y = generate_dataset(100, b, a)
ibelow = np.where(y==-1)
iabove = np.where(y==1)


# plt.scatter(X[ibelow,1], X[ibelow,2])
# plt.scatter(X[iabove,1], X[iabove,2])
# plt.show()

class LogisticModel:
    
    def fit(self, X, y):
        eta = 0.01
        N = X.shape[0]
        w = np.zeros(X.shape[1])
        prev_w = np.ones(X.shape[1])
        epoch = 0
        
        while True:
            prev_w = w
            for n in np.random.permutation(N):
                n = np.random.randint(0, N)
                gradE = -y[n]*X[n,:] / (1 + np.exp(y[n] * w.dot(X[n,:])))
                w = w - eta * gradE

            
            epoch += 1
            if np.linalg.norm(w - prev_w) < 0.01:
                self.w = w
                break
                
        return epoch
            
    
    def predict(self, X):
        return np.sign(X @ self.w)
    
    def cross_entropy_error(self, X, y):
        N = X.shape[0]
        errs = []
        for n in range(N):
            errs.append(np.log(1 + np.exp(-y[n] * self.w.dot(X[n,:]))))
        return np.mean(errs)
    
num_epochs = []
cross_entropy_errors = []
        
for experiment in range(1, 100):
    
    b, a = random_line()
    X, y = generate_dataset(100, b, a)
    lm = LogisticModel()
    epochs = lm.fit(X, y)

    X_test, y_test = generate_dataset(2000, b, a)
    y_predicted = lm.predict(X_test)


    err = lm.cross_entropy_error(X_test, y_test)
    num_epochs.append(epochs)
    cross_entropy_errors.append(err)
    
    print("Epochs: ", np.mean(num_epochs))
    print("Error: ", np.mean(cross_entropy_errors))
    
    
# ibelow = np.where(y_predicted==-1)
# iabove = np.where(y_predicted==1)
# plt.scatter(X_test[ibelow,1], X_test[ibelow,2])
# plt.scatter(X_test[iabove,1], X_test[iabove,2])
# plt.title("Predicted")
# plt.show()

# ibelow = np.where(y_test==-1)
# iabove = np.where(y_test==1)
# plt.scatter(X_test[ibelow,1], X_test[ibelow,2])
# plt.scatter(X_test[iabove,1], X_test[iabove,2])
# plt.title("Actual")
# plt.show()

# %%
