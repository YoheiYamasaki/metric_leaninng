import numpy as np
import maxflow

from matplotlib import pyplot as plt
import networkx as nx

def plot_graph_2d(graph, nodes_shape, plot_weights=True, plot_terminals=True, font_size=7):
    X, Y = np.mgrid[:nodes_shape[0], :nodes_shape[1]]
    aux = np.array([Y.ravel(), X[::-1].ravel()]).T
    positions = {i: v for i, v in enumerate(aux)}
    positions['s'] = (-1, nodes_shape[0] / 2.0 - 0.5)
    positions['t'] = (nodes_shape[1], nodes_shape[0] / 2.0 - 0.5)

    nxgraph = graph.get_nx_graph()
    if not plot_terminals:
        nxgraph.remove_nodes_from(['s', 't'])

    plt.clf()
    nx.draw(nxgraph, pos=positions)

    if plot_weights:
        edge_labels = {}
        for u, v, d in nxgraph.edges(data=True):
            edge_labels[(u,v)] = d['weight']
        nx.draw_networkx_edge_labels(nxgraph,
                                     pos=positions,
                                     edge_labels=edge_labels,
                                     label_pos=0.3,
                                     font_size=font_size)

def create_graph():
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((5,5))
    
    # Edges pointing backwards (left, left up and left down) with infinite
    # capacity
    structure = np.array([[np.inf, 0, 0],
                          [np.inf, 0, 0],
                          [np.inf, 0, 0]
                         ])
    g.add_grid_edges(nodeids, structure=structure, symmetric=False)
    
    # Set a few arbitrary weights
    weights = np.array([[100, 110, 120, 130, 140]]).T + np.array([0, 2, 4, 6, 8])
    print(weights)

    # Edges pointing right
    structure = np.zeros((3,3))
    structure[1,2] = 1
    g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)
    
    # Edges pointing up
    structure = np.zeros((3,3))
    structure[0,1] = 1
    g.add_grid_edges(nodeids, structure=structure, weights=weights+100, symmetric=False)
    print("structure", structure)

    # Edges pointing down
    structure = np.zeros((3,3))
    structure[2,1] = 1
    g.add_grid_edges(nodeids, structure=structure, weights=weights+200, symmetric=False)
   

    print("structure", structure)
    # Source node connected to leftmost non-terminal nodes.
    left = nodeids[:, 0]
    g.add_grid_tedges(left, np.inf, 0)
    # Sink node connected to rightmost non-terminal nodes.
    right = nodeids[:, -1]
    g.add_grid_tedges(right, 0, np.inf)
    
    return nodeids, g

if __name__ == '__main__':
    nodeids, g = create_graph()
    print(g)
    
    # plot_graph_2d(g, nodeids.shape)
    
    g.maxflow()
    print(g.get_grid_segments(nodeids))
