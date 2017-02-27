import numpy as np

class Node(object):
    def  __init__(self, inbound_nodes=[]):
        # Nodes from which this node receives values
        self.inbound_nodes = inbound_nodes
        # Nodes to which this node passes values
        self.outbound_nodes = []
        # A calculated value
        self.value = None
        # Gradients
        self.gradients = {}
        
        # For each inbound node, add this node as outbound node
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}

        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1

class Add(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        """
        Forward propagation.

        """
        self.value = 0
        for n in self.inbound_nodes:
            self.value += n.value

class Mul(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        """
        Forward propagation.

        """
        self.value = 1
        for n in self.inbound_nodes:
            self.value *= n.value

class Linear(Node):
    def __init__(self, X, W, B):
        Node.__init__(self, [X, W, B])

    def forward(self):
        """
        Forward propagation.

        """
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        B = self.inbound_nodes[2].value

        self.value = np.dot(X, W) + B

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)

class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def sigmoid(self, X):
        return 1. / (1. + np.exp(-X))

    def forward(self):
        self.value = self.sigmoid(self.inbound_nodes[0].value)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            sigmoid_value = self.sigmoid(self.inbound_nodes[0].value)
            self.gradients[self.inbound_nodes[0]] += sigmoid_value * (1 - sigmoid_value) * grad_cost

class MSE(Node):
    def __init__(self, Y, A):
        Node.__init__(self, [Y, A])

    def forward(self):
        Y = self.inbound_nodes[0].value.reshape(-1, 1)
        A = self.inbound_nodes[1].value.reshape(-1, 1)

        self.m = Y.shape[0]
        self.diff = Y - A
        self.value = 1. / self.m * np.sum(np.square(self.diff))

    def backward(self):
        self.gradients[self.inbound_nodes[0]] = (2. / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2. / self.m) * self.diff


def topological_sort(feed_dict):
    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]

    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)

    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            if len(G[m]['in']) == 0:
                S.add(m)

    return L

def forward_and_backward(graph):
    # Forward pass
    for n in graph:
        n.forward()
    # Backward pass
    for n in graph[::-1]:
        n.backward()

def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t]

def forward_pass(output_node, sorted_nodes):
    """
    Network forward propagation.

    Performs a forward pass through a list of sorted nodes.

    Arguments:
        'output_node': The output node in the graph.
        'sorted_nodes': A topologically sorted list of nodes.

    Returns the value of the output node.
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value
