import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# further reading https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

# y = mx + c
m = -5
c = -100

my_feature = [n for n in range(1, 10)]
my_label = [m*n + c for n in range(1, 10)]

learning_rate = 0.05
bias_learning_rate = 0.5

bias_weight = 0.5
weight = 0.05

we = []
bw = []

for epoch in range(20):
    for n, feat in enumerate(my_feature):

        we.append(weight)
        bw.append(bias_weight)

        inputs = np.array([feat, 1])
        net = inputs.dot(np.array([weight, bias_weight]))

        # chain rule
        # dE/dw = dE/d_net * d_net/d_w
        # dE/d_net = (net - prediction)
        # d_net/d_w = input

        dE_dw = (net - my_label[n])*inputs[0]
        weight = weight - learning_rate*dE_dw

        dE_db = (net - my_label[n]) * 1
        bias_weight = bias_weight - bias_learning_rate*dE_db

print("FINAL WEIGHT:", weight)
print('FINAL BIAS WEIGHT:', bias_weight)

zip_iter = iter(zip(we, bw))
num = iter(n for n in range(len(we)))

fig = plt.figure(1)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)


def animate(i):
    try:
        weight, bias_weight = next(zip_iter)
        n = next(num)
    except StopIteration:
        return

    ax1.clear()
    x1 = np.linspace(-5, 5, 100)
    y1 = m * x1 + c
    ax1.plot(x1, y1, '--r', label=f'y={m}x+{c}', linewidth=5)

    x2 = np.linspace(-5, 5, 100)
    y2 = weight * x2 + bias_weight
    ax1.plot(x2, y2, '-g', label='linear regression', linewidth=3)
    ax1.legend(loc='best')
    plt.grid()

    ax2.clear()
    ax2.plot([n for n in range(n)], we[:n], label='feature weight')
    ax2.plot([n for n in range(n)], bw[:n], label='bias weight')
    ax2.legend(loc='best')
    plt.grid()


ani = animation.FuncAnimation(fig, animate, interval=100)
plt.show()
