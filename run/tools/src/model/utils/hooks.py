from collections import OrderedDict
import torch
from pytorch_lightning.callbacks import Callback


class OptimizerResumeHook(Callback):
    def on_train_start(self, trainer, pl_module):
        for optimizer in trainer.optimizers:
            optimizer.param_groups[0]["capturable"] = True


class SaveIntermediateHook:
    """This is used to get intermediate values in forward() pass."""

    def __init__(self, model, target_device="cpu"):
        named_modules = model.named_modules()
        self.module_names = OrderedDict()
        for name, module in named_modules:
            self.module_names[module] = name
            module.register_forward_hook(self)
        self.device = target_device
        self.reset()

    def reset(self):
        self.values = []

    def __call__(self, module, args, return_val):
        layer_name = self.module_names[module]
        try:
            args = [x.detach().clone().to(device=self.device) for x in args]
        except:
            pass
        if isinstance(return_val, torch.Tensor):
            return_val = return_val.detach().clone().to(device=self.device)
        self.values.append(
            {"layer_name": layer_name, "input": args, "output": return_val}
        )

    def get_module_names(self):
        return [x for x in self.module_names.values()]

    def get_saved_names(self):
        return [x["layer_name"] for x in self.values]

    def get_inputs(self):
        return [
            {"layer_name": x["layer_name"], "input": x["input"]} for x in self.values
        ]

    def get_outputs(self):
        return [
            {"layer_name": x["layer_name"], "output": x["output"]} for x in self.values
        ]

    def is_identity(self, x, y):
        return len(x.flatten()) == len(y.flatten()) and torch.all(
            x.flatten() == y.flatten()
        )

    def is_relu_output(self, x):
        return torch.all(x >= 0.0).item()

    def get_intermediates(self):
        intermediates = OrderedDict()
        for v in self.values:
            for i, x in enumerate([v["input"] + v["output"]]):
                is_return_value = i == len(v.values()) - 2
                key = v["layer_name"] + (".out" if is_return_value else ".in")
                is_unique = True
                for n, y in intermediates.items():
                    if self.is_identity(x, y):
                        print(f"{key} and {n} are equal, omitting {key}")
                        is_unique = False
                        break
                if is_unique:
                    assert key not in intermediates
                    intermediates[key] = x
        return intermediates
