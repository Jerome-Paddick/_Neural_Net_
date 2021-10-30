import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# further reading https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

# y = mx + c
m = 5000
c = 0

my_feature = [n for n in range(1, 10)]
my_label = [m*n + c for n in range(1, 10)]
label_name = f'y={m}x+{c}'

# input variable
my_feature = ([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0])
# thing were predicting
my_label   = ([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])
label_name = 'custom data'

epochs = 40

learning_rate = 0.008
bias_learning_rate = 0.04

bias_weight = 0.5
weight = 0.05

we = []
bw = []
er = []

for epoch in range(epochs):
    costs = []

    for n, feat in enumerate(my_feature):

        we.append(weight)
        bw.append(bias_weight)

        inputs = np.array([feat, 1])
        net = inputs.dot(np.array([weight, bias_weight]))

        error = net - my_label[n]
        costs.append(error)

        # chain rule
        # dE/dw = dE/d_net * d_net/d_w
        # dE/d_net = (net - prediction)
        # d_net/d_w = input

        dE_dw = error*inputs[0]
        weight = weight - learning_rate*dE_dw

        dE_db = (net - my_label[n]) * 1
        bias_weight = bias_weight - bias_learning_rate*dE_db

    # mean squared error
    mse = (1/len(my_feature))*(sum(np.array(costs)**2))
    er.append(mse)

print("FINAL WEIGHT:", weight)
print('FINAL BIAS WEIGHT:', bias_weight)

zip_iter = iter(zip(we, bw))
num = iter(n for n in range(len(we)))
scale = len(we) / len(er)

fig = plt.figure(1, figsize=(8,8))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)


def animate(i):
    try:
        weight, bias_weight = next(zip_iter)
        n = next(num)
    except StopIteration:
        return

    ax1.clear()
    x = np.linspace(min(my_feature), max(my_feature), 100)
    y = weight * x + bias_weight
    ax1.plot(x, y, '-g', label='linear regression', linewidth=3)
    ax1.scatter(my_feature, my_label, c='r', label=label_name, marker='x')
    ax1.set_ylim([min(my_label)-2, max(my_label)+2])
    ax1.legend(loc='best')
    plt.grid()

    ax2.clear()
    ax2.plot([n for n in range(n)], we[:n], label='feature weight')
    ax2.plot([n for n in range(n)], bw[:n], label='bias weight')
    ax2.legend(loc='best')
    plt.grid()

    ax3.clear()
    ax3.plot([n for n in range(int(n/scale))], er[:int(n/scale)], label='mean squared error')
    ax3.legend(loc='best')
    if er[int(n/scale)] < 3:
        ax3.set_ylim([0, 10])
    if er[int(n / scale)] < 0.3:
        ax3.set_ylim([0, 1])
    plt.grid()


ani = animation.FuncAnimation(fig, animate, interval=10)
plt.show()
