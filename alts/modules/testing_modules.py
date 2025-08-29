from typing import Callable

from alts.core.configuration import init
from alts.core.experiment import Experiment
from alts.modules.evaluator import Evaluate, Evaluator
    
class ACEEvaluator(Evaluator):
    """
    ACEEvaluator(func, pre, wrap, post)
    | **Description**
    |   This evaluator does arbitrary code execution whenever the given function is called.
    :param func_path: What function triggers the arbitrary code
    :type func_path: Callable
    :param pre: What function to call before original call
    :type pre: Callable
    :param wrap: What function to handle original call
    :type wrap: Callable
    :param post: What function to call after original call
    :type post: Callable
    """
    #func_path: strNone
    func_path: str
    pre: Callable
    wrap: Callable
    post: Callable

    func: Callable

    def register(self, experiment: Experiment):
        """
        register(self, experiment) -> None
        | **Description**
        |   Modifies the experiment to print new queries before adding them to the experiment's query queue.
        |   Requires the experiment's oracle to be a POracles.

        :param experiment: The experiment to be evaluated
        :type experiment: Experiment
        :raises: TypeError if self.experiment.oracles is not a POracles
        """
        if self.func_path is None:
            raise ValueError("ACEEvaluator: No target function is given")
        
        super().register(experiment)

        loc_func_path = self.func_path.split(".")
        obj = self.experiment
        for i in range(len(loc_func_path)-1):
            obj = getattr(obj, loc_func_path[i])
        if isinstance(getattr(obj,loc_func_path[-1]), Callable):
            setattr(obj, loc_func_path[-1], Evaluate(getattr(obj, loc_func_path[-1]))) 
            self.func = getattr(obj,loc_func_path[-1])
        else:
            raise TypeError(f"Selected object {obj} is not a function")

        if not self.pre is None:
            self.func.pre(self.pre)
        if not self.wrap is None:
            getattr(obj, loc_func_path[-1]).wrap(self.wrap)
        if not self.post is None:
            self.func.post(self.post)

    def __init__(self,*,func_path="",pre=None,wrap=None,post=None):
        self.func_path = func_path # type: ignore
        self.pre = pre # type: ignore
        self.wrap = wrap # type: ignore
        self.post = post # type: ignore
        super().__init__()

class ResultEvaluator(Evaluator):
    """
    ResultEvaluator(shape, func, idx)
    | **Description**
    |   This evaluator keeps track of the chosen function's output.
    :param func: What functions output object to compare
    :type func: Callable
    """
    func_path: str

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
        if self.func_path is None:
            raise ValueError("ShapeEvaluator: No target function is given")
        
        super().register(experiment)

        loc_func_path = self.func_path.split(".")
        obj = self.experiment
        for i in range(len(loc_func_path)-1):
            obj = getattr(obj, loc_func_path[i])
        if isinstance(getattr(obj,loc_func_path[-1]), Callable):
            setattr(obj, loc_func_path[-1], Evaluate(getattr(obj, loc_func_path[-1]))) 
            self.func = getattr(obj,loc_func_path[-1])
        else:
            raise TypeError(f"Selected object {obj} is not a function")
        
        def test_result(func, *args, **kwargs):
            obj_result_constrain = obj.result_constrain() # type: ignore
            results = func(*args, **kwargs)
            print("Experiment."+".".join(loc_func_path)+f": Expected Shape {obj_result_constrain.shape} and got {results[1].shape}")
            return results


        getattr(obj, loc_func_path[-1]).wrap(test_result)
    
    def __init__(self,*,func_path=""):
        self.func_path = func_path # type: ignore
        super().__init__()