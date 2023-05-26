from typing import List
import logging
import torch
import torch._dynamo as dynamo

# torch._dynamo.config.log_level = logging.ERROR
# dynamo.config.verbose = True
# dynamo.config.output_graph_code = True

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable

@dynamo.optimize(my_compiler)
def foo(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

foo(torch.randn(10), torch.randn(10))



