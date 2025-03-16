from attr import dataclass


TREE_UNDEFINED = -2


def prune_node_from_tree(tree, node_id):
    """Prune a node from a tree by setting the children to -1"""
    if tree.children_left[node_id] != -1:
        prune_node_from_tree(tree, tree.children_left[node_id])
        prune_node_from_tree(tree, tree.children_right[node_id])
    tree.children_left[node_id] = -1
    tree.children_right[node_id] = -1
    tree.feature[node_id] = TREE_UNDEFINED


def is_leaf(tree, node):
    return tree.tree_.children_left[node] == -1


def post_order_traversal(tree, node, traversal, depth):
    """Saves the post traversal order in traversal. Each entry is formatted (node, depth)"""
    if node != -1:
        post_order_traversal(tree, tree.tree_.children_left[node], traversal, depth + 1)
        post_order_traversal(
            tree, tree.tree_.children_right[node], traversal, depth + 1
        )
        if not is_leaf(tree, node):
            traversal.append((node, depth))


def get_max_depth(tree, node, depth=0):
    if node == -1:
        return depth
    return max(
        get_max_depth(tree, tree.tree_.children_left[node], depth + 1),
        get_max_depth(tree, tree.tree_.children_right[node], depth + 1),
    )


@dataclass
class TreeTraversalElement:
    tree: str
    node_id: int
    num_samples: int
    impurity: float
    depth: int


def get_traversal_order(trees, sort_key=lambda elem: elem.num_samples, decending=False):
    traversal_list = []
    for tree_name, tree in trees.items():
        traversal = []
        post_order_traversal(tree, 0, traversal, 0)
        for node_id, depth in traversal:
            traversal_list.append(
                TreeTraversalElement(
                    tree=tree_name,
                    node_id=node_id,
                    num_samples=tree.tree_.n_node_samples[node_id],
                    impurity=tree.tree_.impurity[node_id],
                    depth=depth,
                )
            )

    return sorted(traversal_list, key=sort_key, reverse=decending)
