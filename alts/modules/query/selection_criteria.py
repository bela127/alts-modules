#Fully documented as of 20.07.2024
"""
:doc:`Core Module </core/query/selection_criteria>`
"""
from alts.core.query.selection_criteria import SelectionCriteria
import numpy as np

class NoSelectionCriteria(SelectionCriteria):
    """
    | **Description**
    |   Gives all queries a score of 0.
    """
    def query(self, queries):
        """
        | **Description**
        |   Gives all queries a score of 0.

        :param queries: A list of queries to evaluate
        :type queries: Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        :return: queries, associated scores (all zeros)
        :rtype: Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_, Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        """
        scores = np.zeros((queries.shape[0],1))
        return queries, scores

class AllSelectionCriteria(SelectionCriteria):
    """
    | **Description**
    |   Gives all queries a score of 1.
    """
    def query(self, queries):
        """
        | **Description**
        |   Gives all queries a score of 1.

        :param queries: A list of queries to evaluate
        :type queries: Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        :return: queries, associated scores (all ones)
        :rtype: Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_, Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        """
        scores = np.ones((queries.shape[0],1))
        return queries, scores

class RandomSelectionCriteria(SelectionCriteria):
    """
    | **Description**
    |   Gives each query a random score.
    """
    def query(self, queries):
        """
        | **Description**
        |   Gives each query a random score from 0 to 1.

        :param queries: A list of queries to evaluate
        :type queries: Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        :return: queries, associated scores
        :rtype: Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_, Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        """
        scores = np.random.uniform(0,1,size=(queries.shape[0],1))
        return queries, scores
    
    #TODO Selection by distance from most similar query