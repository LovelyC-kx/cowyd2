"""Inference-only repairs for the original SJ .14 ta/ca/sa architecture.

Does not import the newer cloud compatibility module or modify global SJ classes.
The checkpoint cannot identify a parameter-free final activation; caller must
explicitly select relu (published forward intent) or identity (historical variant).
"""
import inspect
import types
import torch
from torch import nn
from spikingjelly.activation_based import layer


def prepare_original(module, state, output_activation):
    if output_activation not in ('relu', 'identity'):
        raise ValueError('Original attention needs explicit --legacy-attention-output relu or identity; weights alone cannot identify this operation')
    if not any('.ta.sharedMLP.' in k for k in state):
        raise ValueError('Expected original ta/ca/sa checkpoint; refusing incompatible attention replacement')
    old = layer.MultiDimensionalAttention
    source = inspect.getsource(old)
    marker = 'class MultiDimensionalAttention(base.MultiStepModule):'
    if marker not in source or 'self.ta = TimeAttention' not in source:
        raise ValueError('Unrecognized SpikingJelly attention source; inspect before adapting')
    # Keep the installed original TA -> CA -> SA forward and Conv3d weights.
    source = source.replace(marker, 'class MultiDimensionalAttention(nn.Module):', 1)
    namespace = dict(vars(inspect.getmodule(old)))
    namespace.update(nn=nn, torch=torch)
    exec(compile(source, '<original_attention_module_repair>', 'exec'), namespace)
    repaired = namespace['MultiDimensionalAttention']

    class RegisteredAttention(repaired):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.relu = nn.ReLU() if output_activation == 'relu' else nn.Identity()

    class TDBN2d(layer.ThresholdDependentBatchNorm2d):
        def _check_input_dim(self, x):
            if x.ndim != 4:
                raise ValueError(f'Expected flattened 4-D BN input, got {x.shape}')

    # Only the imported original model sees these aliases.
    proxy = types.SimpleNamespace(**vars(layer))
    proxy.MultiDimensionalAttention = RegisteredAttention
    proxy.ThresholdDependentBatchNorm2d = TDBN2d
    module.layer = proxy
    if hasattr(module, 'MultiDimensionalAttention'):
        module.MultiDimensionalAttention = RegisteredAttention
    if hasattr(module, 'ThresholdDependentBatchNorm2d'):
        module.ThresholdDependentBatchNorm2d = TDBN2d
