"""Read-only preflight: refuse timing/profiling when authorized GPU1 is busy."""
import csv
import io
import json
import subprocess
import time


GPU_UUID = "GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2"


def main():
    samples = []
    for sample in range(2):
        if sample:
            time.sleep(1)
        output = subprocess.check_output([
            "nvidia-smi", "-i", GPU_UUID,
            "--query-gpu=timestamp,uuid,utilization.gpu,memory.used,clocks.sm",
            "--format=csv,noheader,nounits",
        ], text=True)
        row, = list(csv.reader(io.StringIO(output)))
        timestamp, uuid, utilization, memory, clock = [item.strip() for item in row]
        assert uuid == GPU_UUID
        samples.append({"timestamp": timestamp, "uuid": uuid,
                        "utilization_pct": int(utilization),
                        "memory_used_mib": int(memory), "sm_clock_mhz": int(clock)})
    print(json.dumps({"gpu_preflight": samples}, indent=2), flush=True)
    if any(s["utilization_pct"] > 10 or s["memory_used_mib"] > 1024 for s in samples):
        raise SystemExit("GPU1 is busy; timing/profiling was not started. No process was modified.")


if __name__ == "__main__":
    main()
