from alts.core.query.selection_criteria import SelectionCriteria
import numpy as np

class NoSelectionCriteria(SelectionCriteria):

    @property
    def query_pool(self):
        return self.exp_modules.oracle_data_pool

    def query(self, queries):
        scores = np.zeros((queries.shape[0],1))
        return queries, scores

class RandomSelectionCriteria(SelectionCriteria):

    @property
    def query_pool(self):
        return self.exp_modules.oracle_data_pool

    def query(self, queries):
        scores = np.random.uniform(0,1,size=(queries.shape[0],1))
        return queries, scores