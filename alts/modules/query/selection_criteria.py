from alts.core.query.selection_criteria import SelectionCriteria
import numpy as np

class NoSelectionCriteria(SelectionCriteria):

    def query(self, queries):
        scores = np.zeros((queries.shape[0],1))
        return queries, scores

class AllSelectionCriteria(SelectionCriteria):

    def query(self, queries):
        scores = np.ones((queries.shape[0],1))
        return queries, scores

class RandomSelectionCriteria(SelectionCriteria):

    def query(self, queries):
        scores = np.random.uniform(0,1,size=(queries.shape[0],1))
        return queries, scores