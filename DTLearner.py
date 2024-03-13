import numpy as np
from operator import itemgetter
from copy import deepcopy

class DTLearner(object):

    def __init__(self, leaf_size=1, verbose=False, tree=None):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.dtree = deepcopy(tree)

    def author(self):
        return ""

    def tree_recursive_func(self, data_x, data_y):
        row_count = data_x.shape[0]
        factor_count = data_x.shape[1]
        # print(data_y)
        # Use mode for Stock indicator
        vals, counts = np.unique(data_y, return_counts=True)
        index = np.argmax(counts)
        mode = vals[index]
        leaf = np.array([-1, mode, np.nan, np.nan])
        # For corner cases, use the median value
        # leaf = np.array([-1, np.median(data_y), np.nan, np.nan])
        # leaf = np.median(data_y)
        # leaf = np.array([-1, np.mean(data_y), np.nan, np.nan])
        # leaf = np.array([-1, 1, 1, 1])
        # print(leaf)
        # leaf = np.bincount(data_y).argmax()
        # If no row, or row count <= leaf size, or all y are same, return leaf
        if row_count == 0:
            return leaf
        elif row_count <= self.leaf_size:
            return leaf
        elif np.all(data_y == data_y[0]):
            return leaf

        factor_index = np.arange(factor_count)
        #Create a array of coorr coeff
        factor_coeff_corr = np.empty(2)
        for factor_n in range(factor_count):
            ce_abs = abs(np.corrcoef(data_x[:, factor_n], data_y)[0, 1])
            # ce_abs = 1
            if np.isnan(ce_abs):
                ce_abs = 0.0
            # factor_coeff_corr.append((factor_n, ce_abs))
            factor_coeff_corr = np.vstack((factor_coeff_corr, (factor_n, ce_abs)))

        factor_coeff_corr = sorted(factor_coeff_corr, key=itemgetter(1), reverse=True)

        # factor_coeff_corr = factor_coeff_corr[1:]
        #Best feature to split on
        factor_coeff_n = 0
        while factor_index.shape[0] > 0:
            best_f = min(factor_coeff_corr[factor_coeff_n][0],factor_index.shape[0]-1)
            # best_f = min(abs(factor_coeff_corr[factor_coeff_n][0]),factor_index.shape[0]-1)
            split_val = np.median(data_x[:, int(best_f)])

            #Boolean array for assigning item to left/right tree
            left_boolean = data_x[:,int(best_f)] <= split_val
            right_boolean = data_x[:,int(best_f)] > split_val

            # If all item are similar, break
            if len(np.unique(left_boolean)) != 1:
                break

            # remove previous best factor/feature from the array
            factor_index = np.delete(factor_index, int(best_f))
            factor_coeff_n += 1

        if len(factor_index) == 0:
            return leaf

        #Call itself
        left_tree = self.tree_recursive_func(data_x[left_boolean], data_y[left_boolean])
        right_tree = self.tree_recursive_func(data_x[right_boolean], data_y[right_boolean])

        if left_tree.ndim == 1:
            rt_pos = 2
        elif left_tree.ndim > 1:
            rt_pos = left_tree.shape[0] + 1
        root = np.array([int(best_f), split_val, 1, rt_pos])

        return np.vstack((root, left_tree, right_tree))

    def add_evidence(self, data_x, data_y):
        """
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        self.dtree = self.tree_recursive_func(data_x, data_y)


    def query(self, points):
        """
        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """

        pred_y = np.empty(1)
        for point in points:
            pred_y = np.vstack((pred_y, self.recursive_query(point, index=0)))

        pred_y = pred_y[1:]  # remove the first empty value when initiating array
        pred_y = pred_y.reshape(1, points.shape[0])
        return np.asarray(pred_y)

    def recursive_query(self, point, index):
        # print('self.dtree\n', self.dtree)
        factor, split_val = self.dtree[index, 0:2]
        # factor = self.dtree[index][0]
        # split_val = self.dtree[index][1]
        if factor == -1:
            return split_val
        elif point[int(factor)] <= split_val:
            y = self.recursive_query(point, index + int(self.dtree[index, 2]))
        else:
            y = self.recursive_query(point, index + int(self.dtree[index, 3]))
        return y
def author():
    return ""