from ast import Tuple
from alts.core.query.query_decider import QueryDecider
import alts.modules.query.query_decider as qdm

from nptyping import NDArray
import numpy as np

#List of query deciders
query_deciders = [
    qdm.AllQueryDecider,
    qdm.NoQueryDecider,
    qdm.ThresholdQueryDecider,
    qdm.TopKQueryDecider
]

#Positive decisiveness of query decider
def positive_decisiveness(qd: QueryDecider):
    if type(qd) == qdm.AllQueryDecider:   
        assert qd.decide(np.array([0]), np.array([[0]]))[0] == True
    elif type(qd) == qdm.NoQueryDecider:
        assert True
    elif type(qd) == qdm.ThresholdQueryDecider:
        assert qd.decide(np.array([0]), np.array([[qd.threshold + 0.01]]))[0] == True
    elif type(qd) == qdm.TopKQueryDecider:
        assert qd.decide(np.array([0]), np.array([[0]]))[0] == True
    else:
        raise ValueError("QueryDecider not found: {}".format(qd))

#Negative decisiveness of query decider
def negative_decisiveness(qd: QueryDecider):
    if type(qd) == qdm.AllQueryDecider:
        assert True
    elif type(qd) == qdm.NoQueryDecider:
        assert qd.decide(np.array([0]), np.array([[0]]))[0] == False
    elif type(qd) == qdm.ThresholdQueryDecider:
        assert qd.decide(np.array([0]), np.array([[qd.threshold]]))[0] == False
    elif type(qd) == qdm.TopKQueryDecider:
        assert True
    else:
        raise ValueError("QueryDecider not found: {}".format(qd))

#Decision test
def right_decision(qd: QueryDecider):
    if type(qd) == qdm.AllQueryDecider:
        #Normal Queries
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[-1], [0], [0.5]]))
        assert (x,set(y)) == (True, set([0, 1, 2]))
        #Consistent returns
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[-1], [0], [0.5]]))
        assert (x,set(y)) == (True, set([0, 1, 2]))
        #Similar Queries
        x,y  = qd.decide(np.array([0, 1, 2]), np.array([[0], [0], [0]]))
        assert (x,set(y)) == (True, set([0, 1, 2]))
        #Single Candidate
        x,y = qd.decide(np.array([0]), np.array([[0]]))
        assert (x,set(y)) == (True, set([0]))
    elif type(qd) == qdm.NoQueryDecider:
        #Normal Query
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[-1], [0], [0.5]]))
        assert  (x,set(y)) == (False, set([]))
        #Consistent retutns
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[-1], [0], [0.5]]))
        assert  (x,set(y)) == (False, set([]))
        #Similar Queries
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[0], [0], [0]]))
        assert  (x,set(y)) == (False, set([]))
        #Single Candidate
        x,y = qd.decide(np.array([0]), np.array([[0]]))
        assert (x,set(y)) == (False, set([]))
    elif type(qd) == qdm.ThresholdQueryDecider:
        #Normal Query
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[qd.threshold], [qd.threshold - 0.1], [qd.threshold + 0.1]]))
        assert (x, set(y)) == (True, set([2]))
        #Consistent returns
        x,y = qd.decide(np.array([0, 1, 2]), np.array([[qd.threshold], [qd.threshold - 0.1], [qd.threshold + 0.1]]))
        assert (x, set(y)) == (True, set([2]))
    elif type(qd) == qdm.TopKQueryDecider:
        #Normal Query
        x,y = qd.decide(np.array([i for i in range(qd.k + 2)]), np.array([[i/2 - 1] for i in range(qd.k + 2)]))
        assert (x,set(y)) == (True, set([i for i in range(2, qd.k + 2)]))
        #Consistent Returns
        x,y = qd.decide(np.array([i for i in range(qd.k + 2)]), np.array([[i/2 - 1] for i in range(qd.k + 2)]))
        assert (x,set(y)) == (True, set([i for i in range(2, qd.k + 2)]))
        #Similar Queries
        assert len(qd.decide(np.array([i for i in range(qd.k + 2)]), np.array([[0] for i in range(qd.k + 2)]))[1].tolist()) == 4
        assert len(set(qd.decide(np.array([i for i in range(qd.k + 2)]), np.array([[0] for i in range(qd.k + 2)]))[1].tolist())) == 4
        #Less than default candidates
        x,y = qd.decide(np.array([0]), np.array([[0]]))
        assert (x,set(y)) == (True, set([0]))
        x,y = qd.decide(np.array([0, 1]), np.array([[0],[1]]))
        assert (x,set(y)) == (True, set([0, 1]))
    else:
        raise ValueError("QueryDecider not found: {}".format(qd))

#Pass Function for random values
def passer_generator(qd: QueryDecider):
    if type(qd) == qdm.AllQueryDecider:
        return lambda candidates, scores, output: True if (output[0], set(output[1])) == (True, set(candidates)) else False
    elif type(qd) == qdm.NoQueryDecider:
        return lambda candidates, scores, output: True if (output[0], set(output[1])) == (False, set([])) else False
    elif type(qd) == qdm.ThresholdQueryDecider:
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
    elif type(qd) == qdm.TopKQueryDecider:
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
                    