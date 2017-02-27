import numpy as np
from miniflow import *

x = Input()
y = Input()
z = Input()

add1 = Add(x, y, z)

feed_dict = {x: 10, y: 5, z: 20}

sorted_nodes = topological_sort(feed_dict)
output = forward_pass(add1, sorted_nodes)

print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))

mul1 = Mul(x, y, z)

sorted_nodes = topological_sort(feed_dict)
output = forward_pass(mul1, sorted_nodes)

print("{} * {} * {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))

add1 = Add(x, y, z)
mul1 = Mul(add1, z)

sorted_nodes = topological_sort(feed_dict)
output = forward_pass(mul1, sorted_nodes)

print("({} + {} + {}) * {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], feed_dict[z], output))

X = Input()
W = Input()
B = Input()

lin1 = Linear(X, W, B)
sig1 = Sigmoid(lin1)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
B_ = np.array([-3., -5])

feed_dict = {X: X_, W: W_, B: B_}

graph = topological_sort(feed_dict)
output = forward_pass(sig1, graph)

print(output)

Y = Input()
A = Input()

cost = MSE(Y, A)

Y_ = np.array([1, 2, 3])
A_ = np.array([4.5, 5, 10])

feed_dict = {Y: Y_, A: A_}

graph = topological_sort(feed_dict)
output = forward_pass(cost, graph)

print(output)
