# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
import numpy as np


def evendist(len):
    return np.array(np.array_split([x/((len**2)-1) for x in range(len**2)], len))


def sigmoid(np_input):
    """Returns list of sigmoid functions for input list"""
    return np.array([[  1 / (1 + np.exp(-z)) for z in np_input   ]])


input_arrays = [[[0.05, 0.10], [0.01, 0.99]]]# , [[0.1, 0.05],[0.02, 0.98]]]

for x in input_arrays:
    x[0] = np.array(x[0])
    x[1] = np.array(x[1])

runcount = 1
b1 = 0.35
b2 = 0.60
Eta = 0.5  # Learning rate

length = len(input_arrays[0])
I_weights = evendist(length)

I_weights = np.array([[0.15, 0.25], [0.20, 0.30]])
H_weights = np.array([[0.40, 0.50], [0.45, 0.55]])

for x in range(runcount):
    for input_array in input_arrays:

        target = input_array[1]
        fwd_I = np.array([input_array[0]]).T

        # --> FORWARD PASS
        out_fwd_H = sigmoid((fwd_I * I_weights).sum(axis=0) + b1)
        out_fwd_O = sigmoid((out_fwd_H * H_weights).sum(axis=0) + b2)
        # <-- FORWARD PASS

        Delta_O = (-(target-out_fwd_O))*(out_fwd_O*(1-out_fwd_O))
        Delta_H = (-(target-out_fwd_H))*(out_fwd_H*(1-out_fwd_H))

        I_weights = I_weights - Eta * fwd_I*(Delta_O*H_weights).sum(axis=1)*out_fwd_H*(1-out_fwd_H)
        H_weights = H_weights - Eta * out_fwd_H * Delta_O

# print("I_weights", I_weights)
# print("H_weights", H_weights)
print("Final Out:", out_fwd_O)




# numpy axis:   down  --> (axis=0) or ndarray[:, 1]
#               right --> (axis=1) or ndarray[1, :]

# Input = [Ia, Ib, Ic]

#   Ia -< H1 -< Ox
#   Ib -< H2 -< Oy
#   Ic -< H3 -< Oz
#   b1    b2

#   Ia x (Wa1 Wa2 Wa3) = net(H1)
#   Ib x (Wb1 Wb2 Wb3) = net(H2)
#   Ic x (Wc1 Wc2 Wc3) = net(H3)

"""
      O1  O2
       ↓   ↓
H1 → wa1 wa2
H2 → wb1 wb2
      t1  t2
"""

# a = np.array([[1, 2, 3]]).T
# b = np.ones((3,3))
"""
   ___               ___
  |. .|             |. .|
  |_c_|             |_-_|
>-|X X|-< [ERROR] >-|X X|-<
   |_|               |_|
   | |               | |
   \ \              / /

"""

"""
for x in range(runcount):
    for input_array in input_arrays:

        target = input_array[1]
        fwd_I = np.array([input_array[0]]).T

        # print(l1_weights)
        # print(l1_weights[1, :])

        # --> FORWARD PASS
        # net_fwd_H = (fwd_I * I_weights).sum(axis=0) + b1
        out_fwd_H = sigmoid((fwd_I * I_weights).sum(axis=0) + b1)

        # net_fwd_O = (out_fwd_H * H_weights).sum(axis=0) + b2
        out_fwd_O = sigmoid((out_fwd_H * H_weights).sum(axis=0) + b2)

        Error = 0.5*np.square(target-out_fwd_O)
        # <-- FORWARD PASS

        # E = Σ 0.5*square(target_x - out_fwd_O)
        # dE/doO = -(target - out_fwd_O)
        # The partial derivative of the logistic function is the output multiplied by 1 minus the output:
        # oOx = sigmoid(nOx)
        # doO/dnoO = oO(1-oO)
        # Delta_O = dE/doO * doO/dnO

        Delta_O = (-(target-out_fwd_O))*(out_fwd_O*(1-out_fwd_O))
        Delta_H = (-(target-out_fwd_H))*(out_fwd_H*(1-out_fwd_H))
        # print("Delta", Delta_H)

        # noO = Σ( oH*W ) + b2
        # dnoO/dW = oH
        # dE/W = dE/doO*doO/dnoO*dnoO/dW = Delta*out

        # effect of change in hidden layer node will propagate to all Output Nodes
        # dE/doH = Σ dEoOx/doH
        # dEoO1/doH1 = dEoO1/dnO1 * dnO1/doH1
        # dEoO/doO = - (t-oO)
        # doO/dnO = nO(1-nO)
        # dnO1 / doH1 = w5
        # dE1_dnO1 = - (t-oO)*nO(1-nO)
        # dE1/doH1 = - (t-oO)*nO(1-nO)*w5
        # print("dE_dH",(Delta_O*H_weights).sum(axis=1))

        # dE/dw = dE/doH * doH/dnH * dnH/dw
        # doH/dnH = oH(1-oH)
        # dnH/dw = i

        # dE_dw = fwd_I*(Delta_O*H_weights).sum(axis=1)*out_fwd_H*(1-out_fwd_H)
        # print(dE_dw)

        I_weights = I_weights - Eta * fwd_I*(Delta_O*H_weights).sum(axis=1)*out_fwd_H*(1-out_fwd_H)
        # print("I_weights", I_weights)

        # H_weights = H_weights - Eta*(np.array(Delta).T)*out_fwd_O
        H_weights = H_weights - Eta * out_fwd_H * Delta_O
        # print("H_weights", H_weights)



        # dE_doH =
        # I_weights = I_weights - Eta * out


        # print("H_weights\n", H_weights)

        # dE/dw5 = dE/doO1 * doO1/dnO1 * dnO1/dw5
        # we know:  Eo1  -> dEo1/doO1
        #           oO1  -> doO1/dnO1
        #           noO1 -> dnO1/dw5


print("Final Out:", out_fwd_O)
"""

