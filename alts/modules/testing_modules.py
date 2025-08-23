from typing import Callable

from alts.core.configuration import init
from alts.core.experiment import Experiment
from alts.modules.evaluator import Evaluate, Evaluator


class ShapeEvaluator(Evaluator):
    """
    ShapeEvaluator(shape, func, idx)
    | **Description**
    |   This evaluator keeps track of the Query Optimizers outgoing query shape.
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
            raise ValueError("ShapeEvaluator: No function is given")
        super().register(experiment)
        self.query_shape = query_shape

        setattr(self.experiment, self.func, Evaluate(getattr(self.experiment, self.func)))
        getattr(self.experiment, self.func).warp(self.test_shape)

    def test_shape(self, func, *args, **kwargs):
        ret = func(*args, **kwargs)
        obj = ret[self.idx]
        assert obj.shape == self.query_shape, f"Shape has to match. Is:{obj.shape}, Should:{self.query_shape}"
        return ret