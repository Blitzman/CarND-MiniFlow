class Node(object):
    def  __init__(self):
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
        if value is not None:
            self.value = value

class Add(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        """
        Forward propagation.

        """
        raise NotImplemented
