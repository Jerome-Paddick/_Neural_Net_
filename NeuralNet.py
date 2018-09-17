import numpy as np

# asdfasdfasdf
# http://neuralnetworksanddeeplearning.com/chap1.html
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

# numpy axis:   down = 0
#               right = 1

# Input = [Ia, Ib, Ic]

#   Ia -< H1 -< Ox
#   Ib -< H2 -< Oy
#   Ic -< H3 -< Oz
#   b1    b2

#   Ia x (Wa1 Wa2 Wa3) = net(H1)
#   Ib x (Wb1 Wb2 Wb3) = net(H2)
#   Ic x (Wc1 Wc2 Wc3) = net(H3)

"""
Ia x-> Wa1 Wa2 
Ib x-> Wb1 Wb2  
sum columns for net(H)

"""


"""
   ___               ___
  |. .|             |. .|
  |_c_|             |_-_|
>-|X X|-< [ERROR] >-|X X|-<
   |_|               |_|
   | |               | |
   \ \              / /

"""


# a = np.array([[1, 2, 3]]).T
# print(a)
# b = np.ones((3,3))
#
# print( a+b )
# b = np.array("")

def evendist(len):
    return np.array(np.array_split([x/((len**2)-1) for x in range(len**2)], len))

def sigmoid(input):
    """Returns list of sigmoid functions for input list"""
    return np.array([[  1 / (1 + np.exp(-z)) for z in input   ]])


input_arrays = [[[0.05, 0.10], [0.01, 0.99]]]
for x in input_arrays:
    x[0] = np.array(x[0])
    x[1] = np.array(x[1])

b1 = 0.35
b2 = 0.60
Eta = 0.5 # Learning rate

length = len(input_arrays[0])
I_weights = evendist(length)

I_weights = np.array([[0.15, 0.25], [0.20, 0.30]])
H_weights = np.array([[0.40, 0.50], [0.45, 0.55]])
# print(I_weights)

for input_array in input_arrays:
    print("H_weights\n", H_weights)
    target = input_array[1]
    fwd_I = np.array([input_array[0]]).T

    # print(l1_weights[:, 1])
    # print(l1_weights[1, :])

    # net_fwd_H = (fwd_I * I_weights).sum(axis=0) + b1
    out_fwd_H = sigmoid((fwd_I * I_weights).sum(axis=0) + b1)
    # print(out_fwd_H)

    # print(net_fwd_in, type(net_fwd_in))
    # print(out_fwd_in, type(out_fwd_in))

    # net_fwd_O = (out_fwd_H * H_weights).sum(axis=0) + b2
    out_fwd_O = sigmoid((out_fwd_H * H_weights).sum(axis=0) + b2)
    # print(out_fwd_O)

    Error = 0.5*np.square(input_array[1]-out_fwd_O)
    # print("Error", Error)

    # d_Error_out = -(input_array[1] - out_fwd_O)
    # d_out_O = out_fwd_O*(1-out_fwd_O)
    # d_net_O = out_fwd_H
    Delta = -(input_array[1] - out_fwd_O)*(out_fwd_O*(1-out_fwd_O))*out_fwd_H
    # print("Delta", Delta)

    # E = Σ 0.5*square(target_x - out_x)
    # dEo1/doO1 = -(t01 - oO1)
    # oOx = sigmoid(nOx)
    # The partial derivative of the logistic function is the output multiplied by 1 minus the output:
    # doOx/dnoOx = oOx(1-oOx)
    # noO = Σ( oOxWx ) + b2
    # dnoO/dWx = oOx
    # Delta = dEo1/dnoO)
    # DE/dwx = Delta*out

    H_weights = H_weights - Eta * out_fwd_H * (-(target - out_fwd_O)) * (out_fwd_O*(1-out_fwd_O))
    print(H_weights)

    # effect of change in hidden layer node will propagate to all Output Nodes
    # dE/doH1 = Σ dEoOx/doH1
    # dEoO1/doH1
    I_weights = I_weights - Eta * out

    # H_weights = H_weights - Eta*(np.array(Delta).T)*out_fwd_O
    # print("H_weights\n", H_weights)

    # dE/dw5 = dE/doO1 * doO1/dnO1 * dnO1/dw5
    # we know:  Eo1  -> dEo1/doO1
    #           oO1  -> doO1/dnO1
    #           noO1 -> dnO1/dw5




