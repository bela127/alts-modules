from alts.core.oracle.augmentation import Augmentation

def test_candidacy_size():
    """
    | **Description**
    |   To pass, the Decider has to return the right list of preferred candidates 3 times.
    """
    ...

def test_decide_yes(aug: Augmentation):
    """
    | **Description**
    |   To pass, the Decider has to pick the right candidates 3 times.
    |   Certain Deciders may automatically pass this test, such as ``NoQueryDecider``.
    """
    ...

def test_decide_no(aug: Augmentation):
    """
    | **Description**
    |   To pass, the Decider has to refuse all given candidates 3 times.
    |   Certain Deciders may automatically pass this test, such as ``AllQueryDecider``.

    """

def test_values():
    """
    | **Description**
    |   To pass, the Decider has to make the right decision 10 times on random inputs.
    """
    ...