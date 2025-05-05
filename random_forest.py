
import numpy as np
from decision_tree import DecisionTree
from collections import Counter


##### CREDIT: Anushka Trehan HW3
class RandomForest:
    def __init__(self, num_trees=10, max_depth=10):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.decision_trees = [DecisionTree(max_depth=self.max_depth) for _ in range(self.num_trees)]
        self.bootstrap_data = []
        self.bootstrap_labels = []

    def generate_bootstrap_sample(self, dataset):
        num_samples = len(dataset)
        sample_indices = np.random.choice(num_samples, size=num_samples, replace=True)
        sample_X = dataset[sample_indices, :-1]
        sample_y = dataset[sample_indices, -1]
        return sample_X, sample_y

    def bootstrapping(self, full_dataset):
        for _ in range(self.num_trees):
            sample_X, sample_y = self.generate_bootstrap_sample(full_dataset)
            self.bootstrap_data.append(sample_X)
            self.bootstrap_labels.append(sample_y)

    def fitting(self, feature_types):
        for i in range(self.num_trees):
            tree = self.decision_trees[i]
            X = self.bootstrap_data[i]
            y = self.bootstrap_labels[i]
            tree.learn(X, y, feature_types)

    def voting(self, X):
        final_predictions = []
        for instance in X:
            predictions = [tree.classify(instance) for tree in self.decision_trees]
            # none values - was getting weird issues with the credit approval dataset? 
            valid_preds = [p for p in predictions if p is not None]
            if valid_preds:
                majority_label = Counter(valid_preds).most_common(1)[0][0]
            else:
                majority_label = 0# fallback if all trees fail (WHAT ABOUT CATEGORICAL ATTRIBUTES)
            final_predictions.append(majority_label)
        return np.array(final_predictions)
