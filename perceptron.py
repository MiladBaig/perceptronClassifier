import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self.activation
        self.weights = None
        self.bias = None

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        msk = np.random.rand(len(X)) < 0.8
        X_train = X[msk]
        X_test = X[~msk]
        y_train = Y[msk]
        y_test = Y[~msk]

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y_train])

        for i in range(self.n_iters):
            mis_classified = []
            for idx, x_i in enumerate(X_train):

                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                if y_predicted != y_[idx]:
                    mis_classified.append([x_i[0], x_i[1]])
                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
                
            mis_classified = np.asanyarray(mis_classified)    
            
            prediction = self.predict(X_test)
            accuracy = round(np.sum(prediction == y_test) / len(y_test) * 100, 2)
            print(f"epoch {i + 1} accuracy: {accuracy}")
            
            self.draw(X, mis_classified)  
            
            if accuracy < 100:
                epoch = input("Do you want antoher epoch?")
                print("\n\n")
                if epoch == '':
                    continue
                else:
                    break
            else:
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def activation(self, x):
        return np.where(x >= 0, 1, 0)
    
    def draw(self, X, mis_classified):
        plt.style.use('fivethirtyeight')
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=Y)
        plt.scatter(mis_classified[:, 0], mis_classified[:, 1], c = 'red')
        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
        x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

        ymin = np.amin(X[:, 1])
        ymax = np.amax(X[:, 1])
        ax.set_ylim([ymin - 3, ymax + 3])
        plt.show()

# Testing
if __name__ == "__main__":

#Generating dataset from multivariate noraml distribution

    mu1 = np.array([1,1])
    cov1 = 4 * np.eye(2)

    mu2 = np.array([8, 12])
    cov2 = 4 * np.eye(2)

    x1 = np.random.multivariate_normal(mu1, cov1, 100)
    x2 = np.random.multivariate_normal(mu2, cov2, 100)

    y1 = np.zeros(len(x1))
    y2 = np.ones(len(x2))

    X = np.concatenate((x1, x2), axis = 0)
    Y = np.concatenate((y1, y2), axis = 0)


    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X, Y)
