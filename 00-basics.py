import torch

# inputs
X = torch.tensor([2, 3, 4, 5], dtype=torch.float).reshape((4, 1))
y = torch.tensor([5, 7, 9, 11], dtype=torch.float).reshape((4, 1))
y = y.reshape(y.shape[0], 1)

# learnable parameters
w = torch.tensor(1., requires_grad=True).reshape((1, 1))
b = torch.tensor(0., requires_grad=True).reshape((1, 1))


def model(X):
    return X @ w + b


def squared_errors(yhat, y):
    return (yhat - y).pow(2).sum()


learning_rate = 0.001

for i in range(10):
    print(i)
    preds = model(X)
    loss = squared_errors(preds, y)

    loss.backward()

    w = w - w.grad * learning_rate
    b = b - b.grad * learning_rate
