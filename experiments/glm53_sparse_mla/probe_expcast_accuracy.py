"""Accuracy-only Torch emulation of direct FP8 probability codes.

Formula: VC-Attention, Li et al., arXiv:2609.15810v1, Eq.7 (CC BY 4.0).
https://arxiv.org/html/2609.15810v1#S3.SS3
This isolates probability approximation using FP32 matrix products. It does
not reproduce CuTe/TC instruction-level accumulation or measure MLA latency.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import benchmark_source as source


GPU_UUID = 'GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2'
METHODS = ('exact_online', 'fp8_448_exact_sum', 'fp8_256_exact_sum',
           'expcast256_decoded_sum', 'expcast256_exact_sum',
           'expcast256_flush_subnormal', 'expcast448_decoded_sum')


def emulate(logits, values, method):
    heads = logits.shape[0]
    maximum = torch.full((heads,), -torch.inf, device=logits.device)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros((heads, 512), device=logits.device)
    scale = 448.0 if '448' in method else 256.0
    subnormal_codes = zero_codes = 0
    for start in range(0, logits.shape[1], 128):
        tile = logits[:, start:start + 128]
        newmax = torch.maximum(maximum, tile.max(dim=-1).values)
        correction = torch.exp2((maximum - newmax) * math.log2(math.e))
        z = (tile - newmax[:, None]) * math.log2(math.e) + math.log2(scale)
        exact = torch.exp2(z)
        if method.startswith('expcast'):
            # Literal Eq.7 at scale256. The scale448 case is our adaptation.
            upper = 126 if scale == 448 else 120
            codes = torch.round(z * 8.0 + 55.65).clamp(0, upper).to(torch.uint8)
            subnormal_codes += int(((codes > 0) & (codes < 8)).sum().item())
            if method.endswith('flush_subnormal'):
                codes = torch.where(codes < 8, torch.zeros_like(codes), codes)
            zero_codes += int((codes == 0).sum().item())
            probability = codes.contiguous().view(torch.float8_e4m3fn).float()
        elif method.startswith('fp8'):
            probability = exact.to(torch.float8_e4m3fn).float()
        else:
            probability = exact
        use_exact_sum = method == 'exact_online' or method.endswith('exact_sum')
        mass = exact if use_exact_sum else probability
        denominator = denominator * correction + mass.sum(dim=-1)
        numerator = numerator * correction[:, None] + probability @ values[start:start + 128]
        maximum = newmax
    return (numerator / denominator[:, None]).to(torch.bfloat16).float(), {
        'subnormal_codes_before_optional_flush': subnormal_codes,
        'zero_codes': zero_codes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        parser.error('select authorized GPU1 by UUID')
    if args.output_json.exists():
        parser.error('choose a fresh output JSON')
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {'purpose': __doc__, 'gpu_uuid': GPU_UUID, 'atol': 0.01, 'rtol': 0.05,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'benchmark_sha256': hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
              'formula_note': 'Eq.7 literal clipping permits codes1..7; separately audit prose-inspired flushing. No constants fitted.',
              'fixtures': []}
    fixtures = [(8192, 3, 1234, False), (8192, 3, 5678, False),
                (1024, 0, 5678, False), (2, 3, 1234, True)]
    with torch.inference_mode():
        for count, chunk, seed, masked in fixtures:
            source.LOCAL_TOKENS = count
            source.make_sparse_indices.__defaults__ = (count, source.TOPK)
            inputs = source.make_inputs(chunk, 0, seed)
            if masked:
                indices = inputs['block_tables'].view(2, -1)
                indices[0, [0, 31, 32, 63, 64, 127, 128, 511, 1023, 2047]] = -1
                indices[1, 129:] = -1
                indices[1, [31, 32, 127]] = -1
                inputs['seq_lens'][1] = 129
            rows = [0, 1] if masked else [0, 1, 31, 127, 255, 511, 512, count - 1]
            reference = source.reference_rows(inputs, rows)
            kv = inputs['kv_cache'].view(-1, 576).float()
            outputs = {m: [] for m in METHODS}
            code_counts = {m: [] for m in METHODS}
            for row in rows:
                n = int(inputs['seq_lens'][row].item())
                slots = inputs['block_tables'][row, 0, :n].long()
                valid = (slots >= 0) & (slots < kv.shape[0])
                selected = kv[slots.clamp(0, kv.shape[0] - 1)]
                logits = (inputs['query'][row, 0].float() @ selected.T) * 0.0625
                logits[:, ~valid] = -torch.inf
                for method in METHODS:
                    actual, counts = emulate(logits, selected[:, :512], method)
                    outputs[method].append(actual)
                    code_counts[method].append(counts)
            fixture = {'local_tokens': count, 'chunk': chunk, 'seed': seed,
                       'masked': masked, 'rows': rows, 'cases': {}}
            for method in METHODS:
                actual = torch.stack(outputs[method])
                error = actual - reference
                bad = ~torch.isclose(actual, reference, atol=0.01, rtol=0.05)
                coords = bad.nonzero()[:10].tolist()
                stats = {'checked_elements': actual.numel(), 'mismatches': int(bad.sum().item()),
                         'max_abs': float(error.abs().max().item()),
                         'relative_rmse': float((error.square().sum() / reference.square().sum()).sqrt().item()),
                         'finite': bool(torch.isfinite(actual).all().item()),
                         'code_counts_by_row': code_counts[method],
                         'examples': [{'row': rows[r], 'head': h, 'channel': c,
                                       'actual': actual[r, h, c].item(),
                                       'reference': reference[r, h, c].item()}
                                      for r, h, c in coords]}
                stats['pass'] = stats['finite'] and stats['mismatches'] == 0
                fixture['cases'][method] = stats
                print(json.dumps({'fixture': [count, chunk, seed, masked], 'method': method,
                                  **{k: v for k, v in stats.items() if k not in ('examples', 'code_counts_by_row')}}), flush=True)
            assert fixture['cases']['exact_online']['pass'], 'FP32 online control failed'
            report['fixtures'].append(fixture)
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(report, indent=2) + '\n')
    # Approximate-method failures are the result of the diagnostic, not waived checks.
    print('DIAGNOSTIC COMPLETE; inspect per-method pass fields; no MLA latency claim', flush=True)


if __name__ == '__main__':
    main()
