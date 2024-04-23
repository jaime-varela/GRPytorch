# GRPytorch

A simple tool for computing objects of interest in general relativity using Automatic differentiation.

See `lib_test.ipynb` for a simple example use case.  To create a metric module one needs a function of the form:

```python
def metric_function(coordinate_vector: torch.Tensor) -> torch.Tensor:
    return torch.diag([-1,1,1,1])
```

See `GRPytorch_metrics.py` for example metrics.  For now only schwarchild was added.


The example in `schwar_test.ipynb` uses Einsteinpy to compare the symbolic output to the numerical estimates obtained via automatic differentiaiton.


## Issues with Einstein Tensor In Vacuum

Due to round-off errors, the Ricci and Einstein tensor in vacuum may not necessarily be zero. I have not figured out a good method to fix  the round-off issue.


