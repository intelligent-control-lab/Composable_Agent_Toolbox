from graphviz import Digraph

g = Digraph('G', filename='cluster_edge.gv')
g.attr(compound='true')

with g.subgraph(name='cluster0') as c:
    c.node("a")
    c.node("b")
    c.node("c")
    c.node("d")

with g.subgraph(name='cluster1') as c:
    c.edges(['eg', 'ef'])

g.edges(['ab', 'ac', 'bd', 'cd'])
g.edge('b', 'f', lhead='cluster1')
g.edge('d', 'e')
g.edge('c', 'g', ltail='cluster0', lhead='cluster1')
g.edge('c', 'e', ltail='cluster0')
g.edge('d', 'h')

g.view()