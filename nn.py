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
