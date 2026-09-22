"""Uninstrumented mixed-MMA validation after matching guarded memory/sync proofs.
No MLA performance claim; this runner does not alter the tested kernel class.
"""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
EXPERIMENT_ROOT = Path(__file__).resolve().parent
if __name__ == '__main__' and '--help' not in sys.argv:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--output-dir', type=Path)
    initial, _ = bootstrap.parse_known_args()
    if initial.output_dir is not None:
        directory = initial.output_dir.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        os.chdir(directory)
        sys.argv += ['--output-dir', str(directory)]
import probe_tmem_mixed_mma as probe_module
from probe_tmem_mixed_mma import MixedMmaProbe, GPU_UUID, torch, cuda, cute, from_dlpack

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks', type=int, choices=(1, 296), default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        parser.error('select the authorized GPU1 UUID')
    if not 1 <= args.repeats <= 10:
        parser.error('repeats must be1..10')
    probe_sha = hashlib.sha256(Path(probe_module.__file__).read_bytes()).hexdigest()
    for check in ('memcheck', 'synccheck'):
        for blocks in (1, 296):
            proof = json.loads((EXPERIMENT_ROOT / 'artifacts' / f'mixed_mma_{check}_{blocks}' / 'result.json').read_text())
            assert proof['guardrails'] and proof['source_sha256'] == probe_sha
            assert len(proof['records']) == 3
            assert all(r['output_mismatches'] == r['score_mismatches'] == 0 for r in proof['records'])
    outdir = args.output_dir.resolve()
    if (outdir / 'result.json').exists():
        parser.error('use a fresh output directory')
    generator = torch.Generator(device='cpu').manual_seed(9187)
    ah = torch.randint(-2, 3, (64, 32), generator=generator).float()
    bh = torch.randint(-2, 3, (512, 32), generator=generator).float()
    a, b = [x.to(device='cuda', dtype=torch.float8_e4m3fn) for x in (ah, bh)]
    output = torch.empty((args.blocks, 64, 512), device='cuda', dtype=torch.float32)
    scores = torch.empty((args.blocks, 64, 66), device='cuda', dtype=torch.float32)
    tensors = [from_dlpack(x, assumed_align=16) for x in (a, b, output, scores)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    options = '--gpu-arch sm_103a --keep-cubin --keep-ptx'
    compiled = cute.compile(MixedMmaProbe(), *tensors, stream, options=options)
    cubins = list(outdir.glob('*.cubin'))
    assert len(cubins) == 1, cubins
    for flag, name in [('--dump-resource-usage', 'resources.txt'), ('--dump-sass', 'sass.txt')]:
        (outdir / name).write_text(subprocess.check_output(['cuobjdump', flag, str(cubins[0])], text=True))
    report = {'purpose': __doc__, 'source_sha256': probe_sha,
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'gpu_uuid': GPU_UUID, 'blocks': args.blocks, 'guardrails': False,
              'compile_only': args.compile_only, 'tmem_columns': 256, 'records': []}
    if not args.compile_only:
        reference = (ah @ bh.T) * 2
        raw_scores = ah @ bh[:64].T
        score_reference = torch.cat((raw_scores,
            raw_scores.reshape(64, 2, 32).max(-1).values), dim=1)
        for repeat in range(args.repeats):
            output.fill_(float('nan'))
            scores.fill_(float('nan'))
            compiled(*tensors, stream)
            torch.cuda.synchronize()
            actual, actual_scores = output.cpu(), scores.cpu()
            mismatches = int((actual != reference.unsqueeze(0)).sum())
            score_mismatches = int((actual_scores != score_reference.unsqueeze(0)).sum())
            report['records'].append({'repeat': repeat, 'output_mismatches': mismatches,
                                      'score_mismatches': score_mismatches,
                                      'output_elements': output.numel(), 'score_elements': scores.numel()})
            assert mismatches == score_mismatches == 0, report['records'][-1]
    (outdir / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
