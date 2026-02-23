from __future__ import annotations
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from typing import Iterable, Optional
    from alts.core.data_process.time_source import TimeSource
    from alts.core.data_process.process import Process
    from alts.core.stopping_criteria import StoppingCriteria
    from alts.core.experiment_modules import ExperimentModules
    from alts.core.evaluator import Evaluator
    from alts.core.oracle.oracles import Oracles
    from alts.core.data.data_pools import DataPools

from alts.core.experiment import Experiment
from alts.modules.evaluator import Evaluate, Evaluator
from alts.core.data.constrains import QueryConstrain, ResultConstrain, QueryConstrained, ResultConstrained

import pytest
    
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
    func_path: str
    pre: Callable
    wrap: Callable
    post: Callable

    func: Callable

    def __init__(self,*,func_path="",pre=None,wrap=None,post=None):
        self.func_path = func_path # type: ignore
        self.pre = pre # type: ignore
        self.wrap = wrap # type: ignore
        self.post = post # type: ignore
        super().__init__()

    def register(self, experiment: Experiment):
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
            raise TypeError(f"Selected object {getattr(obj,loc_func_path[-1])} is not a function")

        if not self.pre is None:
            self.func.pre(self.pre)
        if not self.wrap is None:
            getattr(obj, loc_func_path[-1]).wrap(self.wrap)
        if not self.post is None:
            self.func.post(self.post)

class ConstraintEvaluator(Evaluator):
    """
    ConstraintEvaluator(func_path, q_index, r_index)
    | **Description**
    |   This evaluator keeps track of the constraints of the chosen function's in- and output
    :param func_path: Path to the function to observe
    :type func: str
    :param q_index: Index of input queries in args
    :type q_index: int | slice
    :param r_index: Index of output results in args
    :type r_index: int | slice
    """
    func_path: str
    q_index: int | slice
    r_index: int | slice

    def __init__(self, func_path=None, q_index=None, r_index=None):
        self.func_path = func_path # type: ignore
        self.q_index = q_index # type: ignore
        self.r_index = r_index # type: ignore
        super().__init__()

    def register(self, experiment: Experiment):
        """
        register(self, experiment) -> None
        | **Description**
        |   Modifies the experiment to check in-/output constraints of the chosen function.

        :param experiment: The experiment to be evaluated
        :type experiment: Experiment
        """
        if self.func_path is None:
            raise ValueError("ConstraintEvaluator: No target function is given")
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
        
        def test_func(func, *args, **kwargs):
            obj_qc: QueryConstrain | None = obj.query_constrain() if isinstance(obj, QueryConstrained) else None
            obj_rc = obj.result_constrain() if isinstance(obj, ResultConstrained) else None
            results = func(*args, **kwargs)
            
            if obj_qc is not None:
                if isinstance(self.q_index, slice) and self.q_index.start == self.q_index.step == self.q_index.stop == None:
                    if not obj_qc.constrains_met(args[0]):
                        print("Experiment."+".".join(loc_func_path)+f": Expected Query Shape {obj_qc.shape} and got {args[0].shape}")
                        pytest.fail(f"Experiment.{'.'.join(loc_func_path)}: Input Queries outside constraints: {args[0].shape} -> {obj_qc.shape}, Constraints: {obj_qc.count}:{obj_qc.matches_count(args[0])}, {obj_qc.shape}:{obj_qc.matches_shape(args[0])}, ranges:{obj_qc.matches_ranges(args[0])}")
                if self.q_index is not None:
                    if not obj_qc.constrains_met(args[0][self.q_index]):
                        print("Experiment."+".".join(loc_func_path)+f": Expected Query Shape {obj_qc.shape} and got {args[0][self.q_index].shape}")
                        pytest.fail(f"Experiment.{'.'.join(loc_func_path)}: Input Queries outside constraints: {args[0][self.q_index].shape} -> {obj_qc.shape}, Constraints: {obj_qc.count}:{obj_qc.matches_count(args[0][self.q_index])}, {obj_qc.shape}:{obj_qc.matches_shape(args[0][self.q_index])}, ranges:{obj_qc.matches_ranges(args[0][self.q_index])}")
                else:
                    print("Experiment."+".".join(loc_func_path)+f": No Queries expected, passed")
                    pass

            if obj_rc is not None:
                if isinstance(self.r_index, slice) and self.r_index.start == self.r_index.step == self.r_index.stop == None:
                    assert obj_rc.constrains_met(results[self.r_index]), f"Experiment.{'.'.join(loc_func_path)}: Output Results outside constraints: {len(results)}, {results.shape}, Constraints: {obj_rc.count}{obj_rc.matches_count(results)}, {obj_rc.shape}{obj_rc.matches_shape(results)}, ranges:{obj_rc.matches_ranges(results)}"
                if self.r_index is not None:
                    assert obj_rc.constrains_met(results[self.r_index]), f"Experiment.{'.'.join(loc_func_path)}: Output Results outside constraints: {len(results[self.r_index])}, {results[self.r_index].shape}, Constraints: {obj_rc.count}:{obj_rc.matches_count(results[self.r_index])}, {obj_rc.shape}:{obj_rc.matches_shape(results[self.r_index])}, ranges:{obj_rc.matches_ranges(results[self.r_index])}"
                else:
                    print("Experiment."+".".join(loc_func_path)+f": No Results expected, passed")
                    pass

            return results

        getattr(obj, loc_func_path[-1]).wrap(test_func)
    
def query_for_members(module, base_class):
    """
    query_for_members(module, base_class) -> NDAerray
    | **Description**
    |   Queries a module for all members that are subclasses of the given base class.   
    :param module: The module to query
    :type module: module
    :param base_class: The base class to search for subclasses of
    :type base_class: type
    :return: An array of all found subclasses
    :rtype: `NDArrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
    """
    import inspect
    import numpy as np
    return np.array(inspect.getmembers(module, lambda x:inspect.isclass(x) and issubclass(x, base_class) and x is not base_class))[...,1]

from dataclasses import dataclass, field
from alts.core.blueprint import Blueprint
from alts.modules.data_process.time_source import IterationTimeSource
from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.query_queue import FCFSQueryQueue
from alts.modules.oracle.data_source import RandomUniformDataSource
from alts.modules.stopping_criteria import TimeStoppingCriteria
from alts.modules.queried_data_pool import FlatQueriedDataPool
from alts.modules.query.query_sampler import UniformQuerySampler
from alts.core.experiment_modules import InitQueryExperimentModules
from alts.core.query.query_selector import ResultQuerySelector
from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.query.query_decider import AllQueryDecider
from alts.core.oracle.oracles import POracles
from alts.core.data.data_pools import ResultDataPools

@dataclass
class TestBlueprint(Blueprint):
    """
    TestBlueprint()
    | **Configuration**
    |   *Repeat:* 1
    |   *Time Source:* IterationTimeSource()
    |   *Oracles:* POracles(process= FCFSQueryQueue())
    |   *DataPools:* ResultDataPools(result= FlatQueriedDataPool())
    |   *Process:* DataSourceProcess(data_source= RandomUniformDataSource())
    |   *StoppingCriteria:* TimeStoppingCriteria(stop_time= 100)
    |   *ExperimentModules:* InitQueryExperimentModules(
    |                           initial_query_sampler = UniformQuerySampler(num_queries=10),
    |                           query_selector=ResultQuerySelector(
    |                               query_optimizer=NoQueryOptimizer(query_sampler=UniformQuerySampler()),
    |                               query_decider=AllQueryDecider(),
    |                           )
    |                       )
    |   *Evaluators:* ()
    """
    repeat: int = 1

    time_source: TimeSource = IterationTimeSource()

    oracles: Oracles = POracles(process = FCFSQueryQueue())

    data_pools: DataPools = ResultDataPools(result=FlatQueriedDataPool())

    process: Process = DataSourceProcess(
        data_source=RandomUniformDataSource()
    )

    stopping_criteria: StoppingCriteria = TimeStoppingCriteria(stop_time=100)


    experiment_modules: ExperimentModules = InitQueryExperimentModules(
        initial_query_sampler = UniformQuerySampler(num_queries=10),
        query_selector=ResultQuerySelector(
            query_optimizer=NoQueryOptimizer(query_sampler=UniformQuerySampler()),
            query_decider=AllQueryDecider(),
            ),
        )

    evaluators: Iterable[Evaluator] = ()
