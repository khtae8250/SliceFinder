"""
    SliceFinder: automatic data slicing tool.

    The goal is to identify large slices that are both significant and
    interesting (e.g., high concentration of errorneous examples) for
    a given model. SliceFinder can be used to validate and debug models 
    and data. 

    Author: Yeounoh Chung (yeounohster@gmail.com)
"""
import time
import pickle
import numpy as np
import pandas as pd
import functools
import copy
import concurrent.futures
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from scipy import stats
from risk_control import *

"""
    Slice is specified with a dictionary that maps a set of attributes 
    and their values. For instance, '1 <= age < 5' is expressed as {'age':[[1,5]]}
    and 'gender = male' as {'gender':[['male']]}
"""
class Slice:
    def __init__(self, filters, data_idx):
        self.filters = filters
        self.data_idx = data_idx
        self.size = len(data_idx)
        self.effect_size = None
        self.metric = None

    def get_filter(self):
        return self.filters

    def set_filter(self, filters):
        self.filters = filters

    def set_metric(self, metric):
        self.metric = metric

    def set_effect_size(self, effect_size):
        self.effect_size = effect_size

    def intersect(self, s):
        ''' intersect with Slice s '''
        for k, v in list(s.filters.items()):
            if k not in self.filters:
                self.filters[k] = v
            else:
                for condition in s.filters[k]:
                    if condition not in self.filters[k]:
                        self.filters[k].append(condition)

        idx = self.data_idx.intersection(s.data_idx)
        self.data_idx = idx
        self.size = len(self.data_idx)

        return True

class SliceFinder:
    def __init__(self, data, n_bin, loss_list):
        self.data = data
        self.n_bin = n_bin
        self.break_idx = 0
        self.break_time = 0
        self.loss_list = np.array(loss_list)
        
    def find_slice(self, k=10, epsilon=0.4, alpha=0.05, degree=2, max_workers=1):
        ''' Find top-k problematic slices '''
        reference = (np.mean(self.loss_list), np.std(self.loss_list), len(self.loss_list))
        slices, uninteresting = [], []
        ai_filtered_slices, ai_rejected = [], []
        for i in range(1,degree+1):
            print('degree %s'%i)
            
            if i == 1:
                candidates = self.slicing()
            else:
                candidates = self.crossing(uninteresting, i)
            
            # effect-size testing
            interesting, uninteresting_ = self.filter_by_effect_size(candidates, reference, epsilon, max_workers)
            slices += interesting
            uninteresting += uninteresting_
            
            # significant testing
            ai_filtered_slices_, ai_rejected_ = self.filter_by_significance(interesting, reference, alpha, k)
            ai_filtered_slices += ai_filtered_slices_
            ai_rejected += ai_rejected_

            if self.break_time == 1:
                break
            else:
                print("degree : %d, recommended slices : %d" % (i, len(ai_filtered_slices_)))


        with open('log/recommend.p', 'wb') as handle:
            pickle.dump(ai_filtered_slices, handle)
        with open('log/uninteresting.p','wb') as handle:
            pickle.dump(slices, handle)

        return ai_filtered_slices[:k]
    
    def slicing(self):
        ''' Generate base slices '''
        X, y = self.data[0], self.data[1]
        n, m = X.shape[0], X.shape[1]

        slices = []
        for col in X.columns:
            uniques, counts = np.unique(X[col], return_counts=True)
            for v in uniques:
                data_idx = X[X[col] == v].index
                s = Slice({col:[[v]]}, data_idx)
                slices.append(s)
    
        return slices

    def crossing(self, slices, degree):
        ''' Cross uninteresting slices together '''
        crossed_slices = []
        for i in range(len(slices)-1):
            for j in range(i+1, len(slices)):
                if len(slices[i].filters) + len(slices[j].filters) == degree:
                    slice_ij = copy.deepcopy(slices[i])
                    slice_ij.intersect(slices[j])
                    crossed_slices.append(slice_ij)
        return crossed_slices

    def filter_by_effect_size(self, slices, reference, epsilon=0.5, max_workers=1):
        ''' Filter slices by the minimum effect size '''
        filtered_slices, rejected = [], []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            batch_jobs = []
            for s in slices:
                if s.size == 0:
                    continue
                batch_jobs.append(executor.submit(self.eff_size_job, s, reference))
            
            for job in concurrent.futures.as_completed(batch_jobs):
                if job.cancelled():
                    continue
                elif job.done():
                    s = job.result()
                    if s.effect_size >= epsilon:
                        filtered_slices.append(s)
                    else:
                        rejected.append(s)
                        
        filtered_slices = sorted(filtered_slices, key=lambda s: (s.size, s.effect_size), reverse=True)
        return filtered_slices, rejected

    def eff_size_job(self, s, reference):        
        m_slice = self.loss_list[list(s.data_idx)]
        eff_size = effect_size(m_slice, reference)
        s.set_metric(np.mean(m_slice))
        s.set_effect_size(eff_size)
        return s
    
    def filter_by_significance(self, slices, reference, alpha, k):
        ''' Return significant slices '''
        ai_filtered_slices, ai_rejected = [], []

        k_star = 0
        ai_alpha = 0
        alpha_wealth = alpha

        for i in range(1, len(slices)+1):
            s = slices[i-1]
            m_slice = self.loss_list[list(s.data_idx)]
            p = t_testing(m_slice, reference)
            s.p_value = p
                
            ai_alpha = alpha_wealth / (1+i-k_star)
            if p <= ai_alpha:
                ai_filtered_slices.append(s)
                alpha_wealth += alpha
                k_star = i
                self.break_idx = self.break_idx + 1
            else:
                ai_rejected.append(s)
                alpha_wealth -= ai_alpha/(1.-ai_alpha)
            
            if self.break_idx == k:
                self.break_time = 1
                break
            
        return ai_filtered_slices, ai_rejected    
