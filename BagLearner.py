import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl

class BagLearner(object):

    def __init__(self, learner, kwargs={},  bags=20, boost=False, verbose=False ):
        self.verbose = verbose
        learners = []
        for i in range(bags):
            learners.append(learner(**kwargs))
        self.learners = learners
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost

    def add_evidence(self, data_x, data_y):
        bag_size = data_x.shape[0]
        for learner in self.learners:
            picks = np.random.choice(bag_size, bag_size,replace= True)
            bag_x = data_x[picks]
            bag_y = data_y[picks]
            learner.add_evidence(bag_x, bag_y)

    def query(self, points):
        pred_y = np.array([learner.query(points) for learner in self.learners])
        # avg = np.mean(pred_y, axis=0)
        vals, counts = np.unique(pred_y, return_counts=True, axis = 0)
        index = np.argmax(counts)
        avg = vals[index]

        res = avg.flatten()
        return res

    def author(self):
        return ""

def author():
  return ''