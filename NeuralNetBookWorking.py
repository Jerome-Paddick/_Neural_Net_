"""
http://neuralnetworksanddeeplearning.com/chap1.html

Chapter 1

Perceptrons = several binary inputs -> single binary output
Σwx == w.x
output =  {0 if w.x + b <= 0} where b = - threshold
          {1 if w.x + b >  0}

i)
  Suppose we take all the weights and biases in a network of perceptrons, and multiply them by a positive constant,
  c>0. Show that the behaviour of the network doesn't change.
________________________________________________________________________________________________________________________
  w.x + b -> cw.x + cb
  cw.x + cb  = c(w.x + b)
  as long as c is a positive constant, it will not effect if output is negative, 0 or positive

ii)
  Suppose we have the same setup as the last problem - a network of perceptrons. Suppose also that the overall input to
  the network of perceptrons has been chosen. We won't need the actual input value, we just need the input to have been
  fixed. Suppose the weights and biases are such that w⋅x+b≠0 for the input x to any particular perceptron in the
  network. Now replace all the perceptrons in the network by sigmoid neurons, and multiply the weights and biases by a
  positive constant c>0. Show that in the limit as c→∞ the behaviour of this network of sigmoid neurons is exactly the
  same as the network of perceptrons. How can this fail when w⋅x+b=0 for one of the perceptrons?
________________________________________________________________________________________________________________________


"""

