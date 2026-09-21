# Measured full-target results

B300 physical GPU1; b8192, H64, D576/512, TopK2048, chunk3. Each row is a paired warm CUDA-event run with 3 warmups and 5 repeats. All listed runs passed the original 8-row FP32-reference check; this is sampled numerical validation. v002/v003 were later found memory-unsafe and are rejected. NCU metrics are from separate profiled invocations.

| Version | Candidate µs | Paired TRT µs | TRT/candidate | Registers | Local read/write sectors (M) | Tensor active | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| v001 | 268827.58 | 1691.78 | 0.0063x | 254 | unmeasured | 0.66% | prototype |
| v002 | 26413.31 | 1692.96 | 0.0641x | 255 | 194.51 / 164.08 | 6.03% | unsafe; rejected |
| v003 | 19413.22 | 1691.81 | 0.0871x | 255 | 195.30 / 138.19 | 8.29% | unsafe; rejected |
| v004 | 15984.67 | 1691.78 | 0.1058x | 253 | 0.00 / 0.00 | 10.22% | prototype |
| v005 | 14399.74 | 1693.66 | 0.1176x | 186 | 0.00 / 0.00 | 11.47% | prototype |
| v006 | 8459.26 | 1690.59 | 0.1999x | 186 | 0.00 / 0.00 | 19.18% | prototype |
| v007 | 7694.37 | 1691.71 | 0.2199x | 255 | 579.86 / 22.97 | 22.51% | prototype |
| v008 | 7083.84 | 1691.01 | 0.2387x | 255 | 373.29 / 11.95 | 23.51% | prototype |
| v009 | 7310.24 | 1691.78 | 0.2314x | 255 | 205.65 / 6.63 | 14.81% | prototype |
| v010 | 5277.73 | 1691.87 | 0.3206x | 128 | 0.00 / 0.00 | 32.25% | prototype |
| v011 | 5798.05 | 1691.90 | 0.2918x | 128 | 2.10 / 0.13 | 19.30% | prototype |
| v012 | 7994.75 | 1693.89 | 0.2119x | 128 | 0.00 / 0.00 | 20.81% | prototype |
| v013 | 4482.37 | 1691.78 | 0.3774x | 126 | 0.00 / 0.00 | 36.69% | prototype |
| v014 | 3567.87 | 1691.81 | 0.4742x | 124 | 0.00 / 0.00 | 43.64% | prototype |
| v015 | 3767.58 | 1691.84 | 0.4491x | 124 | 0.00 / 0.00 | 27.47% | prototype |
| v016 | 2996.35 | 1691.78 | 0.5646x | 207 | 0.00 / 0.00 | 34.30% | prototype |
| v017 | 4062.53 | 1691.78 | 0.4164x | 202 | 0.00 / 0.00 | 26.87% | prototype |
| v018 | 3589.12 | 1693.79 | 0.4719x | 200 | 0.00 / 0.00 | 30.69% | prototype |
| v019 | 3022.98 | 1691.87 | 0.5597x | 207 | 0.00 / 0.00 | 34.01% | prototype |
| v020 | 3011.68 | 1691.87 | 0.5618x | 130 | 0.00 / 0.00 | 34.20% | prototype |
| v023 | 3019.84 | 1691.94 | 0.5603x | 121 | 0.00 / 0.00 | 17.04% | prototype |
| v024 | 8754.30 | 1691.58 | 0.1932x | 130 | 0.00 / 0.00 | 11.73% | prototype |
| v025 | 3482.69 | 1692.19 | 0.4859x | 130 | 0.00 / 0.00 | 30.26% | prototype |
| v026 | 3114.18 | 1691.84 | 0.5433x | 130 | 0.00 / 0.00 | 33.11% | prototype |
| v027 | 3458.37 | 1691.84 | 0.4892x | 121 | 0.00 / 0.00 | 15.37% | prototype |
| v028 | 3247.42 | 1689.98 | 0.5204x | 168 | 0.00 / 0.00 | 34.06% | prototype |
| v029 | 3284.10 | 1690.02 | 0.5146x | 128 | 0.00 / 0.00 | 33.73% | prototype |
| v030 | 3041.60 | 1691.90 | 0.5563x | 126 | 0.00 / 0.00 | 33.83% | prototype |
| v031 | 2994.08 | 1691.46 | 0.5649x | 192 | 0.00 / 0.00 | 34.41% | prototype |
| v032 | 3031.30 | 1691.65 | 0.5581x | 255 | 30.41 / 13.27 | 33.98% | prototype |
| v033 | 8477.82 | 4419.74 | 0.5213x | 255 | 33.69 / 18.08 | 34.90% | non-isolated timing; do not rank |
| v034 | 2878.69 | 1693.82 | 0.5884x | 90 | 0.00 / 0.00 | 35.80% | prototype |
| v035 | 3045.25 | 1691.90 | 0.5556x | 126 | 0.00 / 0.00 | 33.83% | prototype |
| v036 | 2926.85 | 1691.74 | 0.5780x | 80 | 8.91 / 13.12 | 35.09% | prototype |
| v037 | 2777.12 | 1691.78 | 0.6092x | 114 | 0.00 / 0.00 | 37.31% | prototype |
| v039 | 2328.74 | 1691.81 | 0.7265x | 118 | 0.00 / 0.00 | 22.32% | prototype |
| v040 | 2326.69 | 1691.78 | 0.7271x | 118 | 0.00 / 0.00 | 22.30% | prototype |
| v041 | 2316.58 | 1691.84 | 0.7303x | 118 | 0.00 / 0.00 | 22.41% | prototype |
| v042 | 2367.33 | 1689.82 | 0.7138x | 168 | 6.29 / 1.58 | 21.98% | prototype |
| v043 | 2712.74 | 1691.68 | 0.6236x | 110 | 0.00 / 0.00 | 20.82% | prototype |
| v044 | 2277.73 | 1691.84 | 0.7428x | 120 | 0.00 / 0.00 | 22.84% | prototype |
| v045 | 1988.80 | 1691.94 | 0.8507x | 122 | 0.00 / 0.00 | 26.33% | prototype |
| v046 | 1976.22 | 1691.87 | 0.8561x | 120 | 0.00 / 0.00 | 26.57% | prototype |
| v047 | 1972.42 | 1691.81 | 0.8577x | 122 | 0.00 / 0.00 | 26.51% | prototype |
| v048 | 1951.87 | 1691.78 | 0.8667x | 126 | 0.00 / 0.00 | 26.88% | prototype |
| v049 | 1902.72 | 1689.82 | 0.8881x | 126 | 0.00 / 0.00 | 27.44% | fails expanded 512-row check |
| v050 | 1908.67 | 1691.49 | 0.8862x | 126 | 0.00 / 0.00 | 27.38% | prototype |
| v051 | 1892.42 | 1691.74 | 0.8940x | 126 | 0.00 / 0.00 | 27.59% | fails expanded 512-row check |
| v052 | 1894.46 | 1691.90 | 0.8931x | 112 | 0.00 / 0.00 | 27.58% | prototype |
| v053 | 2316.42 | 1687.81 | 0.7286x | 101 | 0.00 / 0.00 | 32.94% | all 8192 rows PASS, seed1234 |
| v054 | 1900.90 | 1691.84 | 0.8900x | 126 | 0.00 / 0.00 | 27.43% | same 9 all-row failures as TRT |
| v055 | 1900.77 | 1691.94 | 0.8901x | 126 | 0.00 / 0.00 | 27.45% | prototype |
| v056 | 1958.11 | 1691.84 | 0.8640x | 96 | 6.29 / 6.40 | 26.55% | prototype |

Raw JSON records exact tensor shapes, seed, software versions, candidate SHA256 and unchanged benchmark SHA256. The baseline B0 used 20 warmups/100 repeats and measured 1860.70 µs warm / 1854.66 µs cold; use the paired baseline for each ratio because clocks vary. No fully validated implementation has yet established a speedup over TRTLLM.

## CUDA Graph validation runs

These are separate warm/cold runs with 20 warmups and 100 repeats; sampled row counts are shown explicitly.

| Version | Checked rows | Cache | Candidate µs | Paired TRT µs | TRT/candidate |
|---|---:|---|---:|---:|---:|
| v013 | 64 | warm | 4482.27 | 1879.84 | 0.4194x |
| v013 | 64 | cold | 4485.31 | 1896.74 | 0.4229x |
| v016 | 64 | warm | 2997.23 | 1873.92 | 0.6252x |
| v016 | 64 | cold | 3000.50 | 1876.53 | 0.6254x |
| v034 | 64 | warm | 2879.57 | 1878.14 | 0.6522x |
| v034 | 64 | cold | 2883.65 | 1870.34 | 0.6486x |
| v037 | 64 | warm | 2777.20 | 1879.97 | 0.6769x |
| v037 | 64 | cold | 2781.15 | 1874.18 | 0.6739x |
| v039 | 64 | warm | 2332.50 | 1872.05 | 0.8026x |
| v039 | 64 | cold | 2338.51 | 1912.77 | 0.8179x |
| v044 | 64 | warm | 2283.34 | 1844.90 | 0.8080x |
| v044 | 64 | cold | 2289.49 | 1896.56 | 0.8284x |
| v045 | 64 | warm | 1992.83 | 1874.18 | 0.9405x |
| v045 | 64 | cold | 1998.91 | 1927.30 | 0.9642x |
| v049 | 64 | warm | 1911.01 | 1875.97 | 0.9817x |
| v049 | 64 | cold | 1911.07 | 1923.18 | 1.0063x |
| v053 | 512 | warm | 2356.21 | 1861.66 | 0.7901x |
| v053 | 512 | cold | 2332.77 | 1939.60 | 0.8315x |
| v054 | 512 | warm | 1912.83 | 1869.98 | 0.9776x |
| v054 | 512 | cold | 1910.88 | 1931.20 | 1.0106x |

See [ITERATIONS.md](ITERATIONS.md) for changes, failed hypotheses, correctness limits and source references.
