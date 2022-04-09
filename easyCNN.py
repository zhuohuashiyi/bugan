# created by zhuohuashiyi in 2022.4.8
# 简单二维卷积神经网络的训练和应用

from mxnet import autograd, nd
from mxnet.gluon import nn


def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i][j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()


if __name__ == "__main__":
    # 以下是卷积运算的应用，用来检测图像变化的边缘(垂直方向)
    # 如果是水平方向上的边缘， 则使用nd.array([[1, -1]]).reshape((2, 1))
    # 如果是对角方向上的边缘， 则使用nd.array(([0, 1, -1, 0)).reshape((2, 2))类似的卷积核
    # 规律就是， 构造一个全零值的卷积核， 将其与图像边缘垂直方向上的值置为1， -1
    X = nd.ones((6, 8))
    X[:, 2: 6] = 0
    print(X)
    K = nd.array(([1, -1, 1, 0])).reshape((2, 2))
    Y = corr2d(X, K)
    print(Y)
    # 以下训练出现问题：当用例和教材不一样后，训练出现loss值增大，结果和实际值也差距较大
    conv2d = nn.Conv2D(1, kernel_size=(2, 2))
    conv2d.initialize()
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 5, 7))
    for i in range(10):
        with autograd.record():
            Y_hat = conv2d(X)
            l = (Y_hat - Y) ** 2
        l.backward()
        conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()

        print("batch %d, loss %.3f" % (i + 1, l.sum().asscalar()))
    print(conv2d.weight.data().reshape(2, 2))