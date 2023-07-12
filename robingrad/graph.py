from graphviz import Digraph
import numpy as np


def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root, filename: str = None, inspect: bool = False, format: str = 'svg', rankdir: str = 'LR'):
    """
    filename: name of the output file to save the DOT file (without extension)
    inspect: flag to include detailed information in the labels
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    for n in nodes:
        if inspect:
            dot.node(name=str(id(n)), label=f"data: {np.round(n.data,4)} | grad: {np.round(n.grad,4)} | shape: {n.shape} | type: {n.data.dtype}", shape='record')
        else:
            dot.node(name=str(id(n)), label=f"name: {n._origin} | shape: {n.shape} | type: {n.data.dtype} | grad: {n.requires_grad}", shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    if filename:
        dot.render(filename, format="png",view=True) 

    return dot