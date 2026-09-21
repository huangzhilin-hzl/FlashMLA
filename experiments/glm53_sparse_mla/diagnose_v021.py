"""Isolate weight-stationary QK before accepting a candidate."""
import torch
import benchmark_source as source
import kernel_v021_diag as kernel
source.LOCAL_TOKENS = 2
source.make_sparse_indices.__defaults__ = (2, source.TOPK)
torch.backends.cuda.matmul.allow_tf32 = False
inputs = source.make_inputs(3, 0, 1234)
out = kernel.make_runner(inputs, 128)()[..., :128].float()
q = inputs['query'].squeeze(1).float()
kv = inputs['kv_cache'].view(-1, 576).float()
idx = inputs['block_tables'].view(2, -1)[:, :128].long()
ref = torch.stack([q[i] @ kv[idx[i]].T for i in range(2)])
print('finite', torch.isfinite(out).sum().item(), 'total', out.numel())
print('max_abs', (out-ref).abs().max().item())
print('first', out[0, :4, :8].tolist())
print('reference', ref[0, :4, :8].tolist())
torch.testing.assert_close(out, ref, atol=0.0001, rtol=0.0001)
print('QK PASS')
