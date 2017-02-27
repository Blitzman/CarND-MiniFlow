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
b = Input()

lin1 = Linear(X, W, b)

feed_dict = {
        X: [6, 14, 3],
        W: [0.5, 0.25, 1.4],
        b: 2
}

graph = topological_sort(feed_dict)
output = forward_pass(lin1, graph)

print(output)
