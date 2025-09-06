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

   

class ResultEvaluator(Evaluator):
    """
    ResultEvaluator(shape, func, idx)
    | **Description**
    |   This evaluator keeps track of the chosen function's output.
    :param func: What functions output object to compare
    :type func: Callable
    """
    func_path: str
    query_index: int
    result_index: int

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
            results = func(*args, **kwargs)
            if self.query_index != None:
                obj_query_constrain = obj.query_constrain() # type: ignore
                print("Experiment."+".".join(loc_func_path)+f": Expected Query Shape {obj_query_constrain.shape} and got {results[self.query_index].shape}")
                assert obj_query_constrain.constrains_met(results[0])
            else:
                print("Experiment."+".".join(loc_func_path)+f": No Query expected, passed")
            if self.result_index != None:
                obj_result_constrain = obj.result_constrain() # type: ignore
                print("Experiment."+".".join(loc_func_path)+f": Expected Result Shape {obj_result_constrain.shape} and got {results[self.result_index].shape}")
                assert obj_result_constrain.constrains_met(results[1])
            else:
                print("Experiment."+".".join(loc_func_path)+f": No Result expected, passed")
            return results


        getattr(obj, loc_func_path[-1]).wrap(test_result)
    
    def __init__(self, func_path="", query_index=None, result_index=None,*args, **kwargs):
        self.func_path = func_path # type: ignore
        self.query_index = query_index # type: ignore
        self.result_index = result_index # type: ignore
        super().__init__()

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
