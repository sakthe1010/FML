import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mse(y_pred, y_true):
    return (np.sum((y_pred - y_true) ** 2)/ len(y_pred))

def analytical_solution(X, y):
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    w = np.array(w)
    w.reshape(-1,1)
    return w

def gradient_descent (X, y, w, learning_rate, epochs, w_ml, error_function):

    for i in range(epochs):
        gradient = (2 * (X.T @ ((X @ w) - y)))
        w = w - learning_rate * gradient
        error_function[i] = np.sum((w - w_ml) ** 2)

    return w


def stochastic_gradient_descent(data, w, learning_rate, epochs, w_ml, error_function):

    for epoch in range(epochs):
        
        # Shuffle the training data
        data.sample(frac=1).reset_index(inplace=True, drop=True)
    
        no_batches = 10
        # Iterate over each batch
        for i in range(0, no_batches):

            st_idx = i*int(len(data)/no_batches)
            end_idx = (i+1)*int(len(data)/no_batches)

            # Separate the features (X_rand) and target (y_rand)
            X_rand = data.iloc[st_idx:end_idx].drop("y", axis=1)
            y_rand = data.iloc[st_idx:end_idx]["y"]
            
            # Calculate the gradient
            gradient = 2 * (X_rand.T @ ((X_rand @ w) - y_rand))
            
            # Update the weights
            w = w - learning_rate * gradient
            
            # Calculate and store the error for the current epoch
            
        error_function[epoch] = np.sum((w - w_ml) ** 2)
    
    return w

def ridge_regresssion(X, y, w, learning_rate, epochs, lamda):
    for i in range(epochs):
        gradient = (2 * (X.T @ ((X @ w) - y)) + 2 * lamda* w) / len(y)
        w = w - learning_rate * gradient

    return w

def guassian_kernel(x, xi, mu):
    z = np.linalg.norm(xi- x)/mu
    return (np.exp((-z**2)/2))/(np.sqrt(2*np.pi))

def kernel_regression(X_train, X_test, mu):
    weights = []
    for i, xi in X_test.iterrows():
        kernels = [guassian_kernel(x, xi, mu) for idx, x in X_train.iterrows()]
        sim_score = [X_train.shape[0]*kernel/np.sum(kernels) for kernel in kernels]
        weights.append(sim_score)
    weights = np.array(weights)
    return weights


# Loading the dataset
train_data = pd.read_csv("FMLA1Q1Data_train.csv")
test_data = pd.read_csv("FMLA1Q1Data_test.csv")
train_data["x0"] = 1
test_data["x0"] = 1

y_train = train_data["y"]
X_train = train_data.drop("y", axis=1)
y_test = test_data["y"]
X_test = test_data.drop("y", axis=1)

#### Analytical Solution
w_ml = analytical_solution(X_train, y_train)
y_pred_ml = X_test @ w_ml
ml_error = mse(y_pred_ml, y_test)
print(f"Analytical error: {ml_error}")

#### Gradient Descent
w_init = np.zeros(X_train.shape[1])
epochs = 20
error_function = np.zeros(epochs)

learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
best_l = 0
weight_error = np.inf
for l in learning_rates :
    w = gradient_descent(X_train, y_train, w_init, l, epochs, w_ml, error_function)
    if mse(w, w_ml) < weight_error :
        weight_error = mse(w, w_ml)
        best_l = l

w = gradient_descent(X_train, y_train, w_init, best_l, epochs, w_ml, error_function)

plt.title("Gradient Descent Error Plot")
plt.xlabel("Epochs")
plt.ylabel("Error b/w analytical and predicted weights")
plt.plot(error_function)
plt.show()

#### Stochastic Gradient Descent    
w_init = np.zeros(X_train.shape[1])
epochs = 20
error_function= np.zeros(epochs)

w_stochastic = stochastic_gradient_descent(train_data, w_init, 0.0001, epochs, w_ml, error_function)

plt.title("Stochiastic Gradient Descent Error Plot")
plt.xlabel("Epochs")
plt.ylabel("Error b/w analytical and predicted weights")
plt.plot(error_function)
plt.show()

### Ridge Regression
# Train test split
train_split_len = int(0.8*len(train_data))
train_split =train_data.iloc[:train_split_len, :]
val_split = train_data.iloc[train_split_len:, :]

X_train_split = train_split.drop("y", axis = 1)
y_train_split = train_split["y"]

X_val_split = val_split.drop("y", axis = 1)
y_val_split = val_split["y"]

w_init = np.zeros(X_train.shape[1])
epochs = 1000
lamda = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
best_lamda = 0

# Note: Im using the earlier found learning rate as the best learning rate.
# Cross validating to find the best lambda. 
lambda_error = np.inf
y_val_pred_ml = X_val_split @ w_ml
for lam in lamda :
    w_ridge = ridge_regresssion(X_train_split, y_train_split, w_init, 0.0001, epochs, lam)
    y_val_pred_ridge = X_val_split @ w_ridge
    if mse(y_val_pred_ml, y_val_pred_ridge) < lambda_error:
        lambda_error = mse(y_val_pred_ml, y_val_pred_ridge)
        best_lamda = lam

w_ridge = ridge_regresssion(X_train_split, y_train_split, w_init, best_l, epochs, best_lamda)
y_pred_ridge = X_test @ w_ridge


print(f"Ridge regression test error: {mse(y_pred_ridge, y_test)}")
print(f"Analytical regression test error: {mse(y_pred_ml, y_test)}")


### Kernel Regression

w_kernel = kernel_regression(X_train, X_test, 0.01)
y_pred_kernel = []
for weight in w_kernel:
    y_pred_kernel.append((weight.T @ y_train)/X_train.shape[0])

kernel_error = (np.sum((y_pred_kernel - y_test)**2))/len(y_test)
print(f"Kernel_error(RBF): {kernel_error}")
