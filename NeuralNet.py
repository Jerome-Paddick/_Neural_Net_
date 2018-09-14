import numpy as np

# numpy axis:   down = 0
#               right = 1

# Input = [Ia, Ib, Ic]

#   Ia -< H1 -< Ox
#   Ib -< H2 -< Oy
#   Ic -< H3 -< Oz
#   b1    b2

#   Ia x (Wa1 Wa2 Wa3) = H1
#   Ib x (Wb1 Wb2 Wb3) = H2
#   Ic x (Wc1 Wc2 Wc3) = H3

"""
   ___               ___
  |. .|             |. .|
  |_c_|             |_-_|
>-|X X|-< [ERROR] >-|X X|-<
   |_|               |_|
   | |               | |
   \ \              / /
   
"""


a = np.array([1, 2, 3])
b = np.array([[]])

def evendist(len):
    return np.array(np.array_split([x/((len**2)-1) for x in range(len**2)], len))

def sig(input):
    """Returns list of sigmoid functions for input list"""
    return [  1 / (1 + np.exp(-z)) for z in input   ]


input_lists = [[1, 10, 100]]
b1 = 0.5
b2 = 0.5
length = len(input_lists[0])
l1_weights = evendist(length)


for input_list in input_lists:
    in_np = np.array([input_list]).T
    # needs to be "[[2 dimentional]]" to transpose
    # in_T = in_np.T

    # print(l1_weights[:, 1])
    # print(l1_weights[1, :])

    # input_layer = np.

    forward_input_layer = (np.multiply(in_np, l1_weights))
    hidden = sig(forward_input_layer.sum(axis=1))
    print(hidden)


    # ran = (np.random.random_sample(size=(1,5)) - 0.5)*2
    # print(ran)




