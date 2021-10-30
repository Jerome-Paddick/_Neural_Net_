import numpy as np

# input variable
my_feature = ([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0])
# thing were predicting
my_label   = ([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

# def calc_mean_sqr_error(X):
#     prediction =
#     mean_squared_error = 1/len(X)


def activation_function(number):
    # logistic function
    return 1/(1 + np.exp(-number))

# i1  -  h1  -  o1
#     X      X
# i2  -  h2  -  o2
#    //     //
# b1     b2

label = np.array([0.01, 0.99])

weights_input_h1 = np.array([0.15, 0.2, 0.35])
weights_input_h2 = np.array([0.25, 0.3, 0.35])
features = np.array([0.05, 0.1, 1])

net_h1 = features.dot(weights_input_h1)
net_h2 = features.dot(weights_input_h2)
out_h1 = activation_function(net_h1)
out_h2 = activation_function(net_h2)
out_h = np.array([out_h1, out_h2, 1])

weights_input_o1 = np.array([0.4, 0.45, 0.6])
weights_input_o2 = np.array([0.5, 0.55, 0.6])
net_o1 = out_h.dot(weights_input_o1)
net_o2 = out_h.dot(weights_input_o2)
out_o1 = activation_function(net_o1)
out_o2 = activation_function(net_o2)

out_o = np.array([out_o1, out_o2])

squared_error = np.sum(0.5*(label - out_o)**2)

dE_do = -(label - out_o)

print(dE_do)
