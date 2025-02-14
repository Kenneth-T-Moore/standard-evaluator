import copy
import pytest
import pandas as pd
import numpy as np
from standard_evaluator import FloatVariable, IntVariable, ArrayVariable
from standard_evaluator import EvaluatorInfo, OptProblem, MAXINT
from standard_evaluator import (
    evaluator_info_to_opt_problem,
    opt_problem_to_evaluator_info,
)

def test_evaluator_opt_problem_converters():
    my_eval = EvaluatorInfo(
        name="dummy",
        inputs=[
            ArrayVariable(name="x1", default=[[299.3, 243.5]]),
            FloatVariable(name="p2", class_type="float"),
        ],
        outputs=[FloatVariable(name="y2", class_type="float")],
    )
    # Convert the evaluator information to an opt problem
    my_opt = evaluator_info_to_opt_problem(my_eval)
    # Convert the opt problem back to an evaluator
    my_new_eval = opt_problem_to_evaluator_info(my_opt)
    np.testing.assert_equal(my_eval.model_dump(), my_new_eval.model_dump())

