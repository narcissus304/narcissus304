import numpy as np


# 激活函数和它的导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# 初始化权重和偏置
np.random.seed(42)
input_size = 3
hidden_size = 4
output_size = 1

W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(output_size)

# 输入数据
X = np.array([[0.1, 0.2, 0.3]])
y_true = np.array([[0.4]])


# 前向传播
def forward(X) -> object:
    global A1, A2
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = Z2
    return A2


# 计算损失
def compute_loss(y_pred, y_true):
    return 0.5 * np.sum((y_pred - y_true) ** 2)


# 反向传播
def backward(X, y_true, learning_rate=0.01):
    global W1, b1, W2, b2
    m = X.shape[0]

    dA2 = A2 - y_true
    dZ2 = dA2
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0) / m

    # 更新权重和偏置
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2


# 训练
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    y_pred = forward(X)
    loss = compute_loss(y_pred, y_true)
    backward(X, y_true, learning_rate)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

print("Final prediction:", forward(X))
