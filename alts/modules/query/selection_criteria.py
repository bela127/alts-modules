#Version 1.1.1 conform as of 29.11.2024
"""
| *alts.modules.query.selection_criteria*
| :doc:`Core Module </core/query/selection_criteria>`
"""
from alts.core.query.selection_criteria import SelectionCriteria
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Tuple
    from nptyping import NDArray, Number, Shape

class NoSelectionCriteria(SelectionCriteria):
    """
    NoSelectionCriteria()
    | **Description**
    |   Gives all queries a score of 0.
    """
    def query(self, queries) -> 'Tuple[NDArray[Shape["query_nr, ... query_dims"], Number], NDArray[Shape["query_nr, [query_score]"], Number]]': # type: ignore
        """
        query(self, queries) -> (queries, scores)
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
    AllSelectionCriteria()
    | **Description**
    |   Gives all queries a score of 1.
    """
    def query(self, queries) -> 'Tuple[NDArray[Shape["query_nr, ... query_dims"], Number], NDArray[Shape["query_nr, [query_score]"], Number]]': # type: ignore
        """
        query(self, queries) -> (queries, scores)
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
    RandomSelectionCriteria()
    | **Description**
    |   Gives each query a random score.
    """
    def query(self, queries) -> 'Tuple[NDArray[Shape["query_nr, ... query_dims"], Number], NDArray[Shape["query_nr, [query_score]"], Number]]': # type: ignore
        """
        query(self, queries) -> (queries, scores)
        | **Description**
        |   Gives each query a random score from 0 to 1.

        :param queries: A list of queries to evaluate
        :type queries: Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        :return: queries, associated scores
        :rtype: Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_, Iterable over `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
        """
        scores = np.random.uniform(0,1,size=(queries.shape[0],1))
        return queries, scores