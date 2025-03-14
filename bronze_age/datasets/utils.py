import networkx as nx
import numpy as np


def randomgraph_dimater(n, min_diameter, max_diameter):

    diameter = np.random.randint(min_diameter, max_diameter)
    g = nx.complete_graph(n)

    count = 0
    # we remove random edges until goal diameter "diameter" or more is reached
    while True:
        current_diameter = nx.diameter(g)
        count += 1

        if count > 1000:
            count = 0
            g = nx.complete_graph(n)

        if current_diameter >= diameter:
            return g

        if n >= 100 and max_diameter > 2:
            repeat = n
        elif n >= 100:
            repeat = 500
        else:
            repeat = 1

        for _ in range(repeat):
            i, j = np.random.permutation(n)[:2]
            if g.has_edge(i, j):
                g.remove_edge(i, j)
                if not nx.is_connected(g):
                    g.add_edge(i, j)


def randomgraph(n, **args):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    tree = set()
    nodes = list(range(n))
    current = np.random.choice(nodes)
    tree.add(current)
    while len(tree) < n:
        nxt = np.random.choice(nodes)
        if not nxt in tree:
            tree.add(nxt)
            g.add_edge(current, nxt)
            g.add_edge(nxt, current)
        current = nxt

    for _ in range(n // 5):
        i, j = np.random.permutation(n)[:2]
        while g.has_edge(i, j):
            i, j = np.random.permutation(n)[:2]
        g.add_edge(i, j)
        g.add_edge(j, i)
    return g


def get_localized_distances(g, n):
    seen = set()
    distances = {}
    queue = [(n, 0)]
    while queue:
        node, distance = queue.pop(0)
        if node in distances and distances[node] < distance:
            continue
        distances[node] = distance
        for nb in g.neighbors(node):
            if nb not in seen:
                seen.add(node)
                queue.append((nb, distance + 1))
    return [distances[i] for i in range(g.number_of_nodes())]


def random_tree(n, seed=None, create_using=None):
    """Returns a uniformly random tree on `n` nodes.

    Parameters
    ----------
    n : int
        A positive integer representing the number of nodes in the tree.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    NetworkX graph
        A tree, given as an undirected graph, whose nodes are numbers in
        the set {0, …, *n* - 1}.

    Raises
    ------
    NetworkXPointlessConcept
        If `n` is zero (because the null graph is not a tree).

    Notes
    -----
    The current implementation of this function generates a uniformly
    random Prüfer sequence then converts that to a tree via the
    :func:`~networkx.from_prufer_sequence` function. Since there is a
    bijection between Prüfer sequences of length *n* - 2 and trees on
    *n* nodes, the tree is chosen uniformly at random from the set of
    all trees on *n* nodes.

    Examples
    --------
    >>> tree = nx.random_tree(n=10, seed=0)
    >>> print(nx.forest_str(tree, sources=[0]))
    ╙── 0
        ├── 3
        └── 4
            ├── 6
            │   ├── 1
            │   ├── 2
            │   └── 7
            │       └── 8
            │           └── 5
            └── 9

    >>> tree = nx.random_tree(n=10, seed=0, create_using=nx.DiGraph)
    >>> print(nx.forest_str(tree))
    ╙── 0
        ├─╼ 3
        └─╼ 4
            ├─╼ 6
            │   ├─╼ 1
            │   ├─╼ 2
            │   └─╼ 7
            │       └─╼ 8
            │           └─╼ 5
            └─╼ 9
    """
    if n == 0:
        raise nx.NetworkXPointlessConcept("the null graph is not a tree")
    # Cannot create a Prüfer sequence unless `n` is at least two.
    if n == 1:
        utree = nx.empty_graph(1, create_using)
    else:
        rng = np.random.RandomState(seed)
        sequence = [rng.choice(range(n)) for i in range(n - 2)]
        utree = nx.from_prufer_sequence(sequence)

    if create_using is None:
        tree = utree
    else:
        tree = nx.empty_graph(0, create_using)
        if tree.is_directed():
            # Use a arbitrary root node and dfs to define edge directions
            edges = nx.dfs_edges(utree, source=0)
        else:
            edges = utree.edges

        # Populate the specified graph type
        tree.add_nodes_from(utree.nodes)
        tree.add_edges_from(edges)

    return tree
