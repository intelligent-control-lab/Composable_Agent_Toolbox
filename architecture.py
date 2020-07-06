import networkx as nx
import matplotlib.pyplot as plt
from grave import plot_network
from grave.style import use_attributes


dependencies:
    "control":
        

def hilighter(event):
    # if we did not hit a node, bail
    if not hasattr(event, 'nodes') or not event.nodes:
        return

    # pull out the graph,
    graph = event.artist.graph

    # clear any non-default color on nodes
    # for node, attributes in graph.nodes.data():
    #     attributes.pop('color', None)

    # for u, v, attributes in graph.edges.data():
    #     attributes.pop('width', None)

    for node in event.nodes:
        if 'color' in graph.nodes[node]:
            graph.nodes[node].pop('color', None)
        else:
            graph.nodes[node]['color'] = 'C1'

        
            else:
                edge_attribute['width'] = 3

    # update the screen
    event.artist.stale = True
    event.artist.figure.canvas.draw_idle()


graph = nx.barbell_graph(5, 7)

fig, ax = plt.subplots()
art = plot_network(graph, ax=ax, node_style=use_attributes(),
                   edge_style=use_attributes())

art.set_picker(10)
ax.set_title('Click on the nodes!')
fig.canvas.mpl_connect('pick_event', hilighter)
plt.show()