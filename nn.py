from miniflow import *

x = Input()
y = Input()
z = Input()

f = Add(x, y, z)

feed_dict = {x: 10, y: 5, z: 20}

sorted_nodes = topological_sort(feed_dict)
output = forward_pass(f, sorted_nodes)

print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))
