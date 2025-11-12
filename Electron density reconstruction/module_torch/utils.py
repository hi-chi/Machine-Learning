import torch


def optimizer_to(optim, device):
    for param in optim.state.values():

        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)

        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def non_zero_norm(tensor):
    max_vals = torch.amax(tensor, dim=tuple(range(1, tensor.dim())), keepdim=True).clamp_min(1e-8)
    return tensor / max_vals