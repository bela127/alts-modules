from alts.core.query.query_decider import QueryDecider
import alts.modules.query.query_decider as qdm

import numpy as np
import pytest

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
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([0, 1, 2]).flat))
        #Consistent returns
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[-1], [0], [0.5]]))
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([0, 1, 2]).flat))
        #Similar Scores
        x,y  = qd.decide(np.array([0, 1, 2]), np.array([[0], [0], [0]]))
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([0, 1, 2]).flat))
        #Single Candidate
        x,y = qd.decide(np.array([0]), np.array([[0]]))
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([0]).flat))
    elif isinstance(qd, qdm.NoQueryDecider):
        #Normal Query
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[-1], [0], [0.5]]))
        assert x == False and np.array_equal(np.sort(y.flat), np.sort(np.array([]).flat))
        #Consistent retutns
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[-1], [0], [0.5]]))
        assert x == False and np.array_equal(np.sort(y.flat), np.sort(np.array([]).flat))
        #Similar Scores
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[0], [0], [0]]))
        assert x == False and np.array_equal(np.sort(y.flat), np.sort(np.array([]).flat))
        #Single Candidate
        x,y = qd.decide(np.array([0]), np.array([[0]]))
        assert x == False and np.array_equal(np.sort(y.flat), np.sort(np.array([]).flat))
    elif isinstance(qd, qdm.ThresholdQueryDecider):
        #Normal Query
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[qd.threshold], [qd.threshold - 0.1], [qd.threshold + 0.1]]))
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([2]).flat))
        #Consistent returns
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[qd.threshold], [qd.threshold - 0.1], [qd.threshold + 0.1]]))
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([2]).flat))
    elif isinstance(qd, qdm.TopKQueryDecider):
        #Normal Query
        x,y = qd.decide(np.array([i for i in range(qd.k + 2)]), np.array([[i/2 - 1] for i in range(qd.k + 2)]))
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([i for i in range(2, qd.k + 2)]).flat))
        #Consistent Returns
        x,y = qd.decide(np.array([i for i in range(qd.k + 2)]), np.array([[i/2 - 1] for i in range(qd.k + 2)]))
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([i for i in range(2, qd.k + 2)]).flat))
        #Similar Scores
        assert len(qd.decide(np.array([i for i in range(qd.k + 2)]), np.array([[0] for i in range(qd.k + 2)]))[1].tolist()) == 4
        queries = []
        for query in qd.decide(np.array([i for i in range(qd.k + 2)]), np.array([[0] for i in range(qd.k + 2)]))[1]:
            queries.append(query[0])
        assert len(set(queries)) == 4
        #Less than default candidates
        x,y = qd.decide(np.array([0]), np.array([[0]]))
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([0]).flat))
        x,y = qd.decide(np.array([0, 1]), np.array([[0],[1]]))
        assert x == True and np.array_equal(np.sort(y.flat), np.sort(np.array([0, 1]).flat))
    else:
        raise ValueError("QueryDecider not found: {}".format(qd))

#Returns the associated pass function
def passer_generator(qd: QueryDecider):
    if isinstance(qd, qdm.AllQueryDecider):
        return lambda candidates, scores, output: True if (output[0], set(output[1])) == (True, set(candidates)) else False
    elif isinstance(qd, qdm.NoQueryDecider):
        return lambda candidates, scores, output: True if (output[0], set(output[1])) == (False, set([])) else False
    elif isinstance(qd, qdm.ThresholdQueryDecider):
        def f(candidates, scores, output):  # type: ignore
            sol = set()
            for i in range(len(candidates)):
                if scores[i][0] > qd.threshold:
                    sol.add(candidates[i])
            if output[0] == bool(set(output[1])) and set(output[1]) == sol:
                return True
            else:
                return False
        return f
    elif isinstance(qd, qdm.TopKQueryDecider):
        def f(candidates, scores, output): #type:ignore
            least_score = min([scores[candidates.tolist().index(x)][0] for x in output[1]])
            if len(output[1]) < qd.k and len(output[1]) != len(candidates) or len(output[1]) > qd.k or output[0] == False:
                return False
            for i in range(len(candidates)):
                if candidates[i] not in output[1] and scores[i][0] > least_score:
                    return False
            return True
        return f
    else:
        raise ValueError("QueryDecider not found: {}".format(qd))