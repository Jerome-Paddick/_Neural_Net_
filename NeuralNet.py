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


# a = np.array([[1, 2, 3]]).T
# print(a)
# b = np.ones((3,3))
#
# print( a+b )
# b = np.array("")

def evendist(len):
    return np.array(np.array_split([x/((len**2)-1) for x in range(len**2)], len))

def sig(input):
    """Returns list of sigmoid functions for input list"""
    return np.array([  1 / (1 + np.exp(-z)) for z in input   ])


input_lists = [[1, 10, 100]]
b1 = 0.35
b2 = 0.60
length = len(input_lists[0])
l1_weights = evendist(length)

li_weights = np.array([[0.15,0.2],[]])


for input_list in input_lists:
    in_np = np.array([input_list]).T
    # needs to be "[[2 dimentional]]" to transpose
    # in_T = in_np.T

    # print(l1_weights[:, 1])
    # print(l1_weights[1, :])

    net_fwd_in = np.multiply(in_np, l1_weights).sum(axis=1) + b1
    out_fwd_in = sig(net_fwd_in)
    print(net_fwd_in, type(net_fwd_in))
    print(out_fwd_in, type(out_fwd_in))




    # ran = (np.random.random_sample(size=(1,5)) - 0.5)*2
    # print(ran)




