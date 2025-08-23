from typing import Callable

from alts.core.configuration import init
from alts.core.experiment import Experiment
from alts.modules.evaluator import Evaluate, Evaluator


class ShapeEvaluator(Evaluator):
    """
    ShapeEvaluator(shape, func, idx)
    | **Description**
    |   This evaluator keeps track of the chosen function's output shape.
    :param shape: Expected shape of array
    :type shape: tuple[int]
    :param func: What functions output object to compare
    :type func: Callable
    :param idx: What index of the object has to have the given shape
    :type idx: tuple[int]
    """
    shape: tuple[int] = init(default=(1,))
    func: str = init(default=None)
    idx: tuple[int] = init(default=(0,))

    def register(self, experiment: Experiment, query_shape: tuple[int]):
        """
        register(self, experiment) -> None
        | **Description**
        |   Modifies the experiment to print new queries before adding them to the experiment's query queue.
        |   Requires the experiment's oracle to be a POracles.

        :param experiment: The experiment to be evaluated
        :type experiment: Experiment
        :raises: TypeError if self.experiment.oracles is not a POracles
        """
        if self.func is None:
            raise ValueError("ShapeEvaluator: No target function is given")
        super().register(experiment)
        self.query_shape = query_shape

        setattr(self.experiment, self.func, Evaluate(getattr(self.experiment, self.func)))
        getattr(self.experiment, self.func).warp(self.test_shape)

    def test_shape(self, func, *args, **kwargs):
        ret = func(*args, **kwargs)
        obj = ret[self.idx]
        assert obj.shape == self.query_shape, f"Shape has to match. Is:{obj.shape}, Should:{self.query_shape}"
        return ret
    
class ACEEvaluator(Evaluator):
    """
    ACEEvaluator(func, pre, warp, post)
    | **Description**
    |   This evaluator does arbitrary code execution whenever the given function is called.
    :param func: What function triggers the arbitrary code
    :type func: Callable
    :param pre: What function to call before original call
    :type pre: Callable
    :param warp: What function to handle original call
    :type warp: Callable
    :param post: What function to call after original call
    :type post: Callable
    """
    func: str = init(default=None)
    pre: Callable = init(default=None)
    warp: Callable = init(default=None)
    post: Callable = init(default=None)

    def register(self, experiment: Experiment, query_shape: tuple[int]):
        """
        register(self, experiment) -> None
        | **Description**
        |   Modifies the experiment to print new queries before adding them to the experiment's query queue.
        |   Requires the experiment's oracle to be a POracles.

        :param experiment: The experiment to be evaluated
        :type experiment: Experiment
        :raises: TypeError if self.experiment.oracles is not a POracles
        """
        if self.func is None:
            raise ValueError("ACEEvaluator: No target function is given")
        
        super().register(experiment)
        setattr(self.experiment, self.func, Evaluate(getattr(self.experiment, self.func)))

        if not self.pre is None:
            getattr(self.experiment, self.func).pre(self.pre)
        if not self.warp is None:
            getattr(self.experiment, self.func).warp(self.warp)
        if not self.post is None:
            getattr(self.experiment, self.func).post(self.post)

        