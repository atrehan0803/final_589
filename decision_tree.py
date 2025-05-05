import numpy as np
from collections import Counter

#used same code as hw 1 but it did not account for numerical attributes so i had to add that
class Node:
    def __init__(self, attribute=None, label=None, branches=None, threshold=None):
        self.attribute = attribute
        self.label = label
        self.threshold = threshold
        self.branches = branches if branches is not None else {}

def entropy(y_vals):
    _, freqs = np.unique(y_vals, return_counts=True)
    probs = freqs / np.sum(freqs)
    log_probs = np.log2(probs)
    return -np.sum(probs * log_probs)

def gain_numeric(X, y, col_idx):
    col = X[:, col_idx]
    sorted_vals = sorted(set(col))
    
    if len(sorted_vals) < 2:
        return 0, None

    possible_thresh = []
    for j in range(len(sorted_vals) - 1):
        avg = (sorted_vals[j] + sorted_vals[j + 1]) / 2
        possible_thresh.append(avg)

    best_gain = -1
    best_t = None

    for t in possible_thresh:
        left_idxs = col <= t
        right_idxs = col > t

        if not left_idxs.any() or not right_idxs.any():
            continue

        left_y = y[left_idxs]
        right_y = y[right_idxs]

        left_ent = entropy(left_y)
        right_ent = entropy(right_y)

        weighted = (len(left_y) / len(y)) * left_ent + (len(right_y) / len(y)) * right_ent
        gain = entropy(y) - weighted

        if gain > best_gain:
            best_gain = gain
            best_t = t

    return best_gain, best_t

def gain_categorical(X, y, col_idx):
    col = X[:, col_idx]
    vals = np.unique(col)
    ent = 0

    for v in vals:
        y_subset = y[col == v]
        prop = len(y_subset) / len(y)
        ent += prop * entropy(y_subset)

    return entropy(y) - ent

class DecisionTree:
    def __init__(self, max_depth=10, min_gain=0.01, criterion="entropy"):
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.criterion = criterion
        self.tree = None

    def learn(self, X, y, feature_types):
        self.tree = self._grow_tree(X, y, feature_types, depth=0)

    def _grow_tree(self, X, y, feature_types, depth):
        #hyper param max depth
        if len(set(y)) == 1 or depth >= self.max_depth:
            most_common = Counter(y).most_common(1)[0][0]
            return Node(label=most_common)

        n_feats = X.shape[1]
        sample_feats = np.random.choice(n_feats, int(np.sqrt(n_feats)), replace=False)

        best_feat = -1
        best_thresh = None
        top_gain = -1

        for feat in sample_feats:
            if feature_types[feat] == 'numerical':
                gain, thresh = gain_numeric(X, y, feat)
            else:
                gain = gain_categorical(X, y, feat)
                thresh = None

            if gain > top_gain:
                top_gain = gain
                best_feat = feat
                best_thresh = thresh

#hyperparam min_gain
        if top_gain < self.min_gain:
            return Node(label=Counter(y).most_common(1)[0][0])

        new_node = Node(attribute=best_feat, threshold=best_thresh)

        if feature_types[best_feat] == 'numerical':
            left_mask = X[:, best_feat] <= best_thresh
            right_mask = X[:, best_feat] > best_thresh
            new_node.branches['<='] = self._grow_tree(X[left_mask], y[left_mask], feature_types, depth + 1)
            new_node.branches['>'] = self._grow_tree(X[right_mask], y[right_mask], feature_types, depth + 1)
        else:
            for val in np.unique(X[:, best_feat]):
                val_idxs = X[:, best_feat] == val
                branch = self._grow_tree(X[val_idxs], y[val_idxs], feature_types, depth + 1)
                new_node.branches[val] = branch

        return new_node

    def classify(self, row):
        node = self.tree
        while node.label is None:
            val = row[node.attribute]
            if isinstance(val, np.ndarray):
                if val.ndim == 0:
                    val = val.item()
                elif val.ndim == 1 and len(val) == 1:
                    val = val[0]

            if node.threshold is not None:
                if val <= node.threshold:
                    node = node.branches.get('<=', Node(label=None))
                else:
                    node = node.branches.get('>', Node(label=None))
            else:
                if val in node.branches:
                    node = node.branches[val]
                else:
                    return None
        return node.label

