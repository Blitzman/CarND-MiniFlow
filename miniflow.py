import numpy as np

class Node(object):
    def  __init__(self, inbound_nodes=[]):
        # Nodes from which this node receives values
        self.inbound_nodes = inbound_nodes
        # Nodes to which this node passes values
        self.outbound_nodes = []
        # A calculated value
        self.value = None
        
        # For each inbound node, add this node as outbound node
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    def forward(self):
        """
        Forward propagation.

        Compute the output value based on the inbound nodes and
        store the result in this node's value.
        """

        raise NotImplemented

class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value:
            self.value = value

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
