import torch

# Fixme add speed of light and other constants for completion
def schwarchild_metric(in_tensor: torch.Tensor, Mass = 1000,
                       speed_of_light=1.0,
                       gravitational_constant=1.0) -> torch.Tensor:
    rs = (2*gravitational_constant*Mass)/(speed_of_light*speed_of_light)
    # Note need to keep intermediate computations so that the computation graph is generated
    g00 = -(1-(rs/in_tensor[1])) * speed_of_light * speed_of_light
    g11 = 1/(1-(rs/in_tensor[1]))
    g22 = in_tensor[1]*in_tensor[1]
    g33 = in_tensor[1]*in_tensor[1]*torch.sin(in_tensor[2])*torch.sin(in_tensor[2])
    # simple 4D tensor
    return torch.diag(torch.stack([g00,g11,g22,g33]))

