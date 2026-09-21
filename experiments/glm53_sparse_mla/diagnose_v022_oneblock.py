"""Check one attention tile to separate PV/epilogue from online correction."""
import torch
import benchmark_source as source
import kernel_v022 as kernel
source.LOCAL_TOKENS = 2
source.make_sparse_indices.__defaults__ = (2, source.TOPK)
torch.backends.cuda.matmul.allow_tf32 = False
inputs = source.make_inputs(3, 0, 1234)
inputs['seq_lens'].fill_(128)
actual = kernel.make_runner(inputs, 128)().float()
reference = source.reference_rows(inputs, [0, 1])
print('max_abs', (actual-reference).abs().max().item())
print('first', actual[0,:4,:8].tolist())
print('reference', reference[0,:4,:8].tolist())
torch.testing.assert_close(actual, reference, atol=0.01, rtol=0.05)
print('ONE TILE PASS')
