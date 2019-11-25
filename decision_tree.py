"""
    CART implementation with min_effect_size
"""
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from scipy import stats
from slice_finder import *
from risk_control import *
import time

class Node:
    def __init__(self, desc, left_group, right_group, init_left_group, init_right_group, depth):
        self.desc = desc
        self.left_group = left_group
        self.right_group = right_group
        self.init_left_group = init_left_group
        self.init_right_group = init_right_group
        self.left_child = None
        self.right_child = None
        self.depth = depth
        self.parent = None
    
    def operator(self):
        if self.parent is None:
            operator = 'root'
        elif self.parent.left_child is self:
            if self.parent.desc[2]==1:
                operator = '=='
            else:
                operator = '<'
        elif self.parent.right_child is self:
            if self.parent.desc[2]==1:
                operator = '!='
            else:
                operator = '>='
        return operator
 

class DecisionTree:
    def __init__(self, data, init_data, loss_list, cg_list):
        self.data = data
        self.init_data = init_data
        self.loss_list = loss_list
        self.cg_list = cg_list
        
        self.columns = list(data[0].columns.values)
        self.reference = (np.mean(self.loss_list), np.std(self.loss_list), len(self.loss_list))

    def fit(self):
        self.root = self.get_split_(self.data, self.init_data, 0)
        return self

    def split_(self, node, depth, max_depth, min_size):
        # check for no split
        if node.left_group.empty or node.right_group.empty:
            return node

        X_left, y_left = self.data[0].loc[node.left_group], self.data[1].loc[node.left_group]
        X_right, y_right = self.data[0].loc[node.right_group], self.data[1].loc[node.right_group]

        init_X_left, init_y_left = self.init_data[0].loc[node.init_left_group], self.init_data[1].loc[node.init_left_group]
        init_X_right, init_y_right = self.init_data[0].loc[node.init_right_group], self.init_data[1].loc[node.init_right_group]
        
        # process left child
        if len(X_left) >= min_size:
            node.left_child = self.get_split_((X_left, y_left), (init_X_left, init_y_left), node.depth+1)
            if node.left_child is not None:
                node.left_child.parent = node
        
        # process right child
        if len(X_right) >= min_size:
            node.right_child = self.get_split_((X_right, y_right), (init_X_right, init_y_right), node.depth+1)
            if node.right_child is not None:
                node.right_child.parent = node
        
        return node

    def test_split_(self, X, y, attr_idx, value, init_X, cg):
        ''' Calculate information gain with a given node '''
        # for categorical value
        if cg == 1:
            left = X[X.iloc[:, attr_idx] == value].index
            right = X[X.iloc[:, attr_idx] != value].index
            init_left = init_X[init_X.iloc[:, attr_idx] == value].index
            init_right = init_X[init_X.iloc[:, attr_idx] != value].index
        # for numerical value
        else:
            left = X[X.iloc[:, attr_idx] < value].index
            right = X[X.iloc[:, attr_idx] >= value].index
            init_left = init_X[init_X.iloc[:, attr_idx] < value].index
            init_right = init_X[init_X.iloc[:, attr_idx] >= value].index

        IG = self.entropy_(y) - len(left)/len(y) * self.entropy_(y[left]) - len(right)/len(y) * self.entropy_(y[right])
        return IG, left, right, init_left, init_right

    def get_split_(self, data, init_data, depth):
        ''' Split a node into two children that minimize impurity '''
        cg_list = self.cg_list
        X, y = data[0], data[1]
        init_X, init_y = init_data[0], init_data[1]
        n_examples, n_features = data[0].shape
        
        IG, left_group, right_group, init_left_group, init_right_group, attr_idx, value  = 0, pd.DataFrame().index, pd.DataFrame().index, pd.DataFrame(), pd.DataFrame(), None, None
        for attr_idx_ in range(n_features): 
            for v in np.unique(X.iloc[:,attr_idx_]):
                IG_, left_, right_, init_left_, init_right_ = self.test_split_(X, y, attr_idx_, v, init_X, cg_list[attr_idx_])
                if IG < IG_:
                    IG, left_group, right_group, init_left_group, init_right_group, attr_idx, value = IG_, left_, right_, init_left_, init_right_, attr_idx_, v
        
        if attr_idx is None:
            return None
        else:
            node = Node((self.columns[attr_idx], value, cg_list[attr_idx]), left_group, right_group, init_left_group, init_right_group, depth) 
        
        return node

    def entropy_(self, y):
        size = len(y)
        classes = np.unique(y)
        entropy = 0.
        for c in classes:
            p = float(np.sum(y == c)) / size             
            entropy += -p * np.log2(p)
        return entropy


    def recommend_slices(self, k=20, min_effect_size=0.3):
        ''' Start from the root slice (i.e., the entire dataset)
        and go down to find the top-k problematic slices in a breadth-first traversal '''
        
        recommendations, rejected, rec = [], [], []
        candidates = [self.root]
        k_ = 0
        depth = 0

        while len(candidates) > 0:
            candidate = candidates.pop(0)
            if depth != candidate.depth:
                rec = sorted(rec, key=lambda x:(x.size, x.eff_size), reverse=True)
                recommendations = recommendations + rec
                if len(recommendations) >= k:
                    break
                depth = depth+1
                rec = []
            
            indices = candidate.left_group.union(candidate.right_group)
            metrics = self.loss_list[list(indices)]
            
            eff_size = effect_size(metrics, self.reference)
            candidate.indices = indices
            candidate.init_indices = candidate.init_left_group.union(candidate.init_right_group)

            if eff_size > min_effect_size:
                candidate.size = len(indices)
                candidate.eff_size = eff_size
                rec.append(candidate)
                k_ += 1
            # breadth first search (prefer more interpretable slices)
            else:
                rejected.append(candidate)
                candidate = self.split_(candidate, 0, 0, 1)
                if candidate.left_child is not None:
                    candidates.append(candidate.left_child)
                if candidate.right_child is not None:
                    candidates.append(candidate.right_child)
            
        return recommendations[:k]
   
