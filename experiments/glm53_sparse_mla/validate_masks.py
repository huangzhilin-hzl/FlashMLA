"""Exercise holes and partial tiles; keep FP32 tolerance and baseline equality explicit."""
import argparse
import importlib
import torch
import benchmark_source as source

parser = argparse.ArgumentParser()
parser.add_argument('--kernel-version', required=True)
args = parser.parse_args()
source.LOCAL_TOKENS = 2
source.make_sparse_indices.__defaults__ = (2, source.TOPK)
torch.backends.cuda.matmul.allow_tf32 = False
inputs = source.make_inputs(3, 0, 1234)
indices = inputs['block_tables'].view(2, -1)
holes = [0, 31, 32, 63, 64, 127, 128, 511, 1023, 2047]
indices[0, holes] = -1
indices[1, 129:] = -1
indices[1, [31, 32, 127]] = -1
inputs['seq_lens'][1] = 129
candidate = importlib.import_module(f'kernel_{args.kernel_version}').make_runner(inputs, 128)().float()
base = importlib.import_module('kernel_v020').make_runner(inputs, 128)().float()
print('candidate_vs_v020', {'unequal': (candidate != base).sum().item(), 'max_abs': (candidate-base).abs().max().item()})
torch.testing.assert_close(candidate, base, atol=0, rtol=0)
q = inputs['query'].squeeze(1).float()
kv = inputs['kv_cache'].view(-1, 576).float()
reference = []
counts = []
for row in range(2):
    slots = indices[row, :int(inputs['seq_lens'][row])]
    slots = slots[slots >= 0].long()
    counts.append(slots.numel())
    selected = kv[slots]
    scores = (q[row] @ selected.T) * 0.0625
    reference.append(torch.softmax(scores, -1) @ selected[:, :512])
reference = torch.stack(reference)
error = candidate-reference
print('valid_counts', counts)
print('fp32_reference', {'max_abs': error.abs().max().item(), 'rmse': error.square().mean().sqrt().item(), 'atol': 0.01, 'rtol': 0.05})
torch.testing.assert_close(candidate, reference, atol=0.01, rtol=0.05)
print('MASK CHECK PASS')
