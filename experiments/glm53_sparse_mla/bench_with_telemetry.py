"""Read GPU1 telemetry around the unchanged benchmark's whole case windows.

Sampling runs in a separate CPU process. It creates no CUDA context and changes
no device settings. Results are diagnostic, not replacements for uninstrumented
timings. Case windows include warmup, capture, event setup and measured calls;
they are not individual kernel intervals. NVML power/utilization have their own
averaging periods, and SM clocks do not identify the Tensor Core boost state.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import runpy
import statistics
import sys
import time


GPU_UUID = 'GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2'


def sample_nvml(stop, ready, output_path, interval_ms):
    import pynvml as nv

    result = {'gpu_uuid': GPU_UUID, 'samples': [], 'field_errors': {}}
    initialized = False
    try:
        nv.nvmlInit()
        initialized = True
        device = nv.nvmlDeviceGetHandleByUUID(GPU_UUID)
        observed = nv.nvmlDeviceGetUUID(device)
        if isinstance(observed, bytes):
            observed = observed.decode()
        assert observed == GPU_UUID
        fields = {
            'sm_clock_mhz': lambda: nv.nvmlDeviceGetClockInfo(device, nv.NVML_CLOCK_SM),
            'memory_clock_mhz': lambda: nv.nvmlDeviceGetClockInfo(device, nv.NVML_CLOCK_MEM),
            'power_usage_mw': lambda: nv.nvmlDeviceGetPowerUsage(device),
            'temperature_c': lambda: nv.nvmlDeviceGetTemperature(device, nv.NVML_TEMPERATURE_GPU),
            'pstate': lambda: nv.nvmlDeviceGetPowerState(device),
            'clock_event_reasons': lambda: nv.nvmlDeviceGetCurrentClocksEventReasons(device),
            'gpu_utilization_pct': lambda: nv.nvmlDeviceGetUtilizationRates(device).gpu,
        }
        ready.send({'ready': True})
        while not stop.is_set():
            row = {'start_monotonic_ns': time.monotonic_ns()}
            for name, query in fields.items():
                if name in result['field_errors']:
                    continue
                try:
                    row[name] = int(query())
                except Exception as error:
                    result['field_errors'][name] = str(error)
            row['end_monotonic_ns'] = time.monotonic_ns()
            result['samples'].append(row)
            stop.wait(interval_ms / 1000)
    except Exception as error:
        result['fatal_error'] = repr(error)
        ready.send({'ready': False, 'error': repr(error)})
    finally:
        if initialized:
            nv.nvmlShutdown()
        Path(output_path).write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--telemetry-json', type=Path, required=True)
    parser.add_argument('--telemetry-interval-ms', type=float, default=10)
    parser.add_argument('--telemetry-sampling', choices=('on', 'off'), default='on')
    parser.add_argument('--telemetry-entrypoint', choices=('bench.py', 'bench_round_robin.py'),
                        default='bench.py')
    args, benchmark_args = parser.parse_known_args()
    if args.telemetry_interval_ms < 5:
        parser.error('telemetry interval must be at least 5 ms')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        parser.error('this diagnostic requires the authorized GPU1 UUID in CUDA_VISIBLE_DEVICES')
    import benchmark_source as source

    args.telemetry_json.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.telemetry_json.with_suffix('.samples.json')
    if args.telemetry_json.exists() or raw_path.exists():
        parser.error('choose fresh telemetry output paths')
    ctx = mp.get_context('spawn')
    stop = ctx.Event()
    sampler = None
    if args.telemetry_sampling == 'on':
        receive, send = ctx.Pipe(duplex=False)
        sampler = ctx.Process(target=sample_nvml,
                              args=(stop, send, str(raw_path), args.telemetry_interval_ms))
        sampler.start()
        if not receive.poll(30):
            sampler.terminate()
            sampler.join()
            raise RuntimeError('NVML sampler did not initialize')
        state = receive.recv()
        if not state['ready']:
            sampler.join()
            raise RuntimeError(state)

    original_measure = source.measure_case
    windows = []

    def measure_case(case, options, flush_buffer):
        window = {'case': case.name, 'cache': 'cold' if flush_buffer is not None else 'warm',
                  'timing': options.timing, 'start_monotonic_ns': time.monotonic_ns()}
        try:
            samples = original_measure(case, options, flush_buffer)
            window['median_us'] = statistics.median(samples)
            window['samples_us'] = samples
            return samples
        finally:
            window['end_monotonic_ns'] = time.monotonic_ns()
            windows.append(window)

    source.measure_case = measure_case
    sys.argv = [args.telemetry_entrypoint, *benchmark_args]
    try:
        runpy.run_path(str(Path(__file__).with_name(args.telemetry_entrypoint)), run_name='__main__')
    finally:
        source.measure_case = original_measure
        if sampler is not None:
            stop.set()
            sampler.join(10)
            if sampler.is_alive():
                sampler.terminate()
                sampler.join()
                raise RuntimeError('NVML sampler did not stop cleanly')
            if sampler.exitcode != 0:
                raise RuntimeError(f'NVML sampler failed: {sampler.exitcode}')
        raw = json.loads(raw_path.read_text()) if sampler is not None else {'samples': []}
        if raw.get('fatal_error'):
            raise RuntimeError(raw['fatal_error'])
        for window in windows:
            selected = [s for s in raw['samples']
                        if window['start_monotonic_ns'] <= s['start_monotonic_ns']
                        and s['end_monotonic_ns'] <= window['end_monotonic_ns']]
            window['telemetry_sample_count'] = len(selected)
            window['telemetry'] = {}
            for name in ('sm_clock_mhz', 'memory_clock_mhz', 'power_usage_mw',
                         'temperature_c', 'pstate', 'gpu_utilization_pct'):
                values = [s[name] for s in selected if name in s]
                if values:
                    window['telemetry'][name] = {'min': min(values),
                        'median': statistics.median(values), 'max': max(values)}
            window['clock_event_reasons_observed'] = sorted({
                s['clock_event_reasons'] for s in selected if 'clock_event_reasons' in s})
        report = {
            'purpose': 'whole-case NVML diagnostic; not per-kernel timing or boost-state proof',
            'sampling': args.telemetry_sampling, 'interval_ms': args.telemetry_interval_ms,
            'entrypoint': args.telemetry_entrypoint,
            'gpu_uuid': GPU_UUID, 'benchmark_args': benchmark_args,
            'benchmark_sha256': hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
            'sampler_field_errors': raw.get('field_errors', {}), 'windows': windows,
            'nvml_reference': 'https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html',
            'limitations': [
                'Windows include warmup, graph capture, event setup and measured calls.',
                'PowerUsage is a one-second average on newer non-GA100 architectures.',
                'Utilization has a device-defined averaging period; polling does not shorten it.',
                'SM clock samples do not directly report the Tensor Core boost state.',
                'Compare sampling on/off to detect sampler perturbation; no clocks are changed.',
            ],
        }
        args.telemetry_json.write_text(json.dumps(report, indent=2) + '\n')
        print('[TELEMETRY]', args.telemetry_json, flush=True)


if __name__ == '__main__':
    main()
