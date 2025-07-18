#Test version 1.2 as of 18.07.2025
from alts.core.query.query_decider import QueryDecider
import alts.modules.query.query_decider as qdm

import numpy as np
import pytest

"""
| **Test aims**
|   The query deciders are tested for:
|   - Accepting the correct queries
|   - Rejecting the correct queries
|   - Doing both at the same time
"""

#List of query deciders
query_deciders = [
    qdm.AllQueryDecider,
    qdm.NoQueryDecider,
    qdm.ThresholdQueryDecider,
    qdm.TopKQueryDecider
]

#Positive decisiveness of query decider
@pytest.mark.parametrize("qd", query_deciders)
def test_positive_decisiveness(qd: QueryDecider):
    """
    | **Description**
    |   To pass, the Decider has to pick the right candidates 3 times.
    |   Certain Deciders may automatically pass this test, such as ``NoQueryDecider``.
    """
    qd = qd()
    if isinstance(qd, qdm.AllQueryDecider):   
        assert qd.decide(np.array([0]), np.array([[0]]))[0] == True
    elif isinstance(qd, qdm.NoQueryDecider):
        assert True
    elif isinstance(qd, qdm.ThresholdQueryDecider):
        assert qd.decide(np.array([0]), np.array([[qd.threshold + 0.01]]))[0] == True
    elif isinstance(qd, qdm.TopKQueryDecider):
        assert qd.decide(np.array([0]), np.array([[0]]))[0] == True
    else:
        raise ValueError("QueryDecider not found: {}".format(qd))

#Negative decisiveness of query decider
@pytest.mark.parametrize("qd", query_deciders)
def test_negative_decisiveness(qd: QueryDecider):
    """
    | **Description**
    |   To pass, the Decider has to refuse all given candidates 3 times.
    |   Certain Deciders may automatically pass this test, such as ``AllQueryDecider``.
    """
    qd = qd()
    if isinstance(qd, qdm.AllQueryDecider):
        assert True
    elif isinstance(qd, qdm.NoQueryDecider):
        assert qd.decide(np.array([0]), np.array([[0]]))[0] == False
    elif isinstance(qd, qdm.ThresholdQueryDecider):
        assert qd.decide(np.array([0]), np.array([[qd.threshold]]))[0] == False
    elif isinstance(qd, qdm.TopKQueryDecider):
        assert True
    else:
        raise ValueError("QueryDecider not found: {}".format(qd))

#Decision test
@pytest.mark.parametrize("qd", query_deciders)
def test_normal_values(qd: QueryDecider):
    """
    | **Description**
    |   To pass, the Decider has to make the right decision on chosen inputs.
    """
    qd = qd()
    if isinstance(qd, qdm.AllQueryDecider):
        #Normal Queries
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[-1], [0], [0.5]]))
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([0, 1, 2]).flat)), "All query candidates should be chosen"
        #Consistent returns
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[-1], [0], [0.5]]))
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([0, 1, 2]).flat)), "All query candidates should be chosen"
        #Similar Scores
        x,y  = qd.decide(np.array([0, 1, 2]), np.array([[0], [0], [0]]))
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([0, 1, 2]).flat)), "All query candidates should be chosen"
        #Single Candidate
        x,y = qd.decide(np.array([0]), np.array([[0]]))
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([0]).flat)), "All query candidates should be chosen"
    elif isinstance(qd, qdm.NoQueryDecider):
        #Normal Query
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[-1], [0], [0.5]]))
        assert x == False and np.array_equal(np.sort(y.flat), np.sort(np.array([]).flat)), "No query candidates should be chosen"
        #Consistent retutns
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[-1], [0], [0.5]]))
        assert x == False and np.array_equal(np.sort(y.flat), np.sort(np.array([]).flat)), "No query candidates should be chosen"
        #Similar Scores
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[0], [0], [0]]))
        assert x == False and np.array_equal(np.sort(y.flat), np.sort(np.array([]).flat)), "No query candidates should be chosen"
        #Single Candidate
        x,y = qd.decide(np.array([0]), np.array([[0]]))
        assert x == False and np.array_equal(np.sort(y.flat), np.sort(np.array([]).flat)), "No query candidates should be chosen"
    elif isinstance(qd, qdm.ThresholdQueryDecider):
        #Normal Query
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[qd.threshold], [qd.threshold - 0.1], [qd.threshold + 0.1]]))
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([2]).flat)), "Only queries with scores above the threshold value should be chosen"
        #Consistent returns
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[qd.threshold], [qd.threshold - 0.1], [qd.threshold + 0.1]]))
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([2]).flat)), "Only queries with scores above the threshold value should be chosen"
    elif isinstance(qd, qdm.TopKQueryDecider):
        #Normal Query
        x,y = qd.decide(np.array([i for i in range(qd.k + 2)]), np.array([[i/2 - 1] for i in range(qd.k + 2)]))
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([i for i in range(2, qd.k + 2)]).flat)), "Only k query candidates should be chosen"
        #Consistent Returns
        x,y = qd.decide(np.array([i for i in range(qd.k + 2)]), np.array([[i/2 - 1] for i in range(qd.k + 2)]))
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([i for i in range(2, qd.k + 2)]).flat)), "Only k query candidates should be chosen"
        #Similar Scores
        assert len(qd.decide(np.array([i for i in range(qd.k + 2)]), np.array([[0] for i in range(qd.k + 2)]))[1].tolist()) == 4, "Only k query candidates should be chosen"
        queries = []
        for query in qd.decide(np.array([i for i in range(qd.k + 2)]), np.array([[0] for i in range(qd.k + 2)]))[1]:
            queries.append(query[0])
        assert len(set(queries)) == 4 , "Only k query candidates should be chosen"
        #Less than default candidates
        x,y = qd.decide(np.array([0]), np.array([[0]]))
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([0]).flat)), "All candidates should be chosen"
        x,y = qd.decide(np.array([0, 1]), np.array([[0],[1]]))
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([0, 1]).flat)), "All candidates should be chosen"
    else:
        raise ValueError("QueryDecider not found: {}".format(qd))