import numpy as np
import pandas as pd
from config import args
from collections import defaultdict


class Graph():
    r"""Helper class to handle graph properties.

    Parameters
    ----------
    vertices : list
        List of nodes.
    """

    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.V = vertices

    def addEdge(self, u, v):
        """Adding edge to graph."""
        self.graph[u].append(v)

    def isCyclicUtil(self, v, visited, recStack):
        """Utility function to return whether graph is cyclic."""
        # Mark current node as visited and
        # adds to recursion stack
        visited[v] = True
        recStack[v] = True

        # Recur for all neighbours
        # if any neighbour is visited and in
        # recStack then graph is cyclic
        for neighbour in self.graph[v]:
            if visited[neighbour] == False:
                if self.isCyclicUtil(neighbour, visited, recStack) == True:
                    return True
            elif recStack[neighbour] == True:
                return True

        # The node needs to be poped from
        # recursion stack before function ends
        recStack[v] = False
        return False

    def isCyclic(self):
        """Returns whether graph is cyclic."""
        visited = [False] * self.V
        recStack = [False] * self.V
        for node in range(self.V):
            if visited[node] == False:
                if self.isCyclicUtil(node, visited, recStack) == True:
                    return True
        return False

    def topologicalSortUtil(self, v, visited, stack):
        """A recursive function used by topologicalSort ."""
        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, v)

    def topologicalSort(self):
        """A sorting function. """
        # Mark all the vertices as not visited
        visited = [False] * self.V
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        return stack


def structural_causal_process(T, noises=None, seed=None):
    # linear & nonlinear function
    def lin_f(x):
        return x

    def nonlin_f(x):
        return (x + 5. * x ** 2 * np.exp(-x ** 2 / 20.))

    # SCM
    # links = {
    #     0: [((0, -1), 0.9, lin_f)],
    #     1: [((1, -1), 0.8, lin_f), ((0, -1), 0.8, lin_f)],
    #     2: [((2, -1), 0.7, lin_f), ((1, -2), 0.6, lin_f), ((3, -2), 0.6, nonlin_f)],
    #     3: [((3, -1), 0.7, lin_f), ((2, -3), -0.5, lin_f)],
    #     4: [((4, -1), 0.7, lin_f), ((3, -3), -0.5, nonlin_f)]
    # }

    # random SCM
    rands = np.random.random(size=15)
    rands[rands < 0.5] -= 1
    links = {
        0: [((0, -1), rands[0], lin_f)],
        1: [((1, -1), rands[3], lin_f), ((0, -1), rands[4], lin_f)],
        2: [((2, -1), rands[6], lin_f), ((1, -2), rands[7], lin_f)],
        3: [((3, -1), rands[9], lin_f), ((2, -3), rands[10], lin_f), ((0, -1), rands[11], lin_f)],
        4: [((4, -1), rands[12], lin_f), ((3, -3), rands[13], lin_f), ((0, -1), rands[14], lin_f)]
    }

    # generate data
    random_state = np.random.RandomState(seed)

    N = len(links.keys())
    if noises is None:
        noises = [random_state.randn for j in range(N)]

    if N != max(links.keys()) + 1 or N != len(noises):
        raise ValueError("links and noises keys must match N.")

    # Check parameters
    max_lag = 0
    contemp_dag = Graph(N)
    for j in range(N):
        for link_props in links[j]:
            var, lag = link_props[0]
            coeff = link_props[1]
            func = link_props[2]
            if lag == 0: contemp = True
            if var not in range(N):
                raise ValueError("var must be in 0..{}.".format(N - 1))
            if 'float' not in str(type(coeff)):
                raise ValueError("coeff must be float.")
            if lag > 0 or type(lag) != int:
                raise ValueError("lag must be non-positive int.")
            max_lag = max(max_lag, abs(lag))

            # Create contemp DAG
            if var != j and lag == 0:
                contemp_dag.addEdge(var, j)

    if contemp_dag.isCyclic() == 1:
        raise ValueError("Contemporaneous links must not contain cycle.")

    causal_order = contemp_dag.topologicalSort()

    transient = int(.2 * T)

    data = np.zeros((T + transient, N), dtype='float32')
    for j in range(N):
        data[:, j] = noises[j](T + transient)

    for t in range(max_lag, T + transient):
        for j in causal_order:
            for link_props in links[j]:
                var, lag = link_props[0]
                coeff = link_props[1]
                func = link_props[2]
                data[t, j] += coeff * func(data[t + lag, var])

    data = data[transient:]

    nonstationary = (np.any(np.isnan(data)) or np.any(np.isinf(data)))

    # network
    net = np.zeros(shape=[N, N, args.max_tau])
    for n, ls in links.items():
        for l in ls:
            net[l[0][0], n, abs(l[0][1]) - 1] = 1

    return pd.DataFrame(data), nonstationary, net


def gen_5vars(length):
    # generate data
    flows = np.random.random(size=[length + 1, 5])
    flows[0, :] = [0.5, 0.5, 0.5, 0.2, 0.4]

    for t in range(len(flows) - 1):
        flows[t + 1, 0] = flows[t, 0] * (4 - 4 * flows[t, 0] - 2 * flows[t, 1] - 0.4 * flows[t, 2])
        flows[t + 1, 1] = flows[t, 1] * (3.1 - 0.31 * flows[t, 0] - 3.1 * flows[t, 1] - 0.93 * flows[t, 2])
        flows[t + 1, 2] = flows[t, 2] * (2.12 + 0.636 * flows[t, 0] + 0.636 * flows[t, 1] - 2.12 * flows[t, 2])
        flows[t + 1, 3] = \
            flows[t, 3] * (3.8 - 0.111 * flows[t, 0] - 0.011 * flows[t, 1] + 0.131 * flows[t, 2] - 3.8 * flows[t, 3])
        flows[t + 1, 4] = \
            flows[t, 4] * (4.1 - 0.082 * flows[t, 0] - 0.111 * flows[t, 1] - 0.125 * flows[t, 2] - 4.1 * flows[t, 4])

    # connections
    net = np.zeros(shape=[5, 5, args.max_tau])
    net[0, 0, 0], net[1, 0, 0], net[2, 0, 0] = 1, 1, 1
    net[0, 1, 0], net[1, 1, 0], net[2, 1, 0] = 1, 1, 1
    net[0, 2, 0], net[1, 2, 0], net[2, 2, 0] = 1, 1, 1
    net[0, 3, 0], net[1, 3, 0], net[2, 3, 0], net[3, 3, 0] = 1, 1, 1, 1
    net[0, 4, 0], net[1, 4, 0], net[2, 4, 0], net[4, 4, 0] = 1, 1, 1, 1

    return pd.DataFrame(flows), net
