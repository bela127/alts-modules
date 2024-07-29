from alts.core.query.query_decider import QueryDecider

from query_decider_data import *

import numpy as np

#TODO Multidimensional testing

TEST_DEPTH = 100
MAX_DATA_SIZE = 20

def decide_yes(qd: QueryDecider):
    """
    | **Description**
    |   To pass, the Decider has to pick the right candidates 3 times.
    |   Certain Deciders may automatically pass this test, such as ``NoQueryDecider``.
    """
    positive_decisiveness(qd)

def decide_no(qd: QueryDecider):
    """
    | **Description**
    |   To pass, the Decider has to refuse all given candidates 3 times.
    |   Certain Deciders may automatically pass this test, such as ``AllQueryDecider``.
    """
    negative_decisiveness(qd)

def normal_values(qd: QueryDecider):
    """
    | **Description**
    |   To pass, the Decider has to make the right decision on chosen inputs.
    """
    right_decision(qd)

def random_values(qd: QueryDecider):
    passer = passer_generator(qd)
    for i in range(TEST_DEPTH):
        candidates = np.array(list(range(np.random.random_integers(1, MAX_DATA_SIZE))))
        scores = np.array([[x] for x in ((np.random.random((len(candidates))) - 0.5) * MAX_DATA_SIZE)])
        assert passer(candidates, scores, qd.decide(candidates, scores))

def test_all():
    for query_decider in query_deciders:
        qd = query_decider()
        decide_yes(qd)
        decide_no(qd)
        normal_values(qd)
        random_values(qd)