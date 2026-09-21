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
| v058 | 1949.73 | 1691.78 | 0.8677x | 96 | 2.10 / 2.13 | 26.67% | prototype |
| v059 | 2580.70 | 1689.86 | 0.6548x | 96 | 18.35 / 29.36 | 21.47% | prototype |
| v060 | 1937.54 | 1691.84 | 0.8732x | 96 | 0.00 / 0.00 | 26.84% | prototype |
| v061 | 1988.83 | 1691.94 | 0.8507x | 128 | 11.03 / 0.40 | 26.51% | prototype |
| v062 | 1904.77 | 1691.94 | 0.8883x | 110 | 0.00 / 0.00 | 27.49% | prototype |
| v063 | 1890.69 | 1691.94 | 0.8949x | 126 | 0.00 / 0.00 | 27.59% | same 9 all-row failures as TRT |
| v064 | 2121.76 | 1693.92 | 0.7984x | 91 | 0.00 / 0.00 | 24.27% | prototype |
| v065 | 2224.06 | 1693.57 | 0.7615x | 102 | 0.00 / 0.00 | 34.39% | bitwise v053 on all8192 rows, seed1234 |
| v066 | 2160.61 | 1691.81 | 0.7830x | 102 | 0.00 / 0.00 | 35.40% | all8192 rows PASS, seeds1234/5678 |
| v067 | 2048.38 | 1693.86 | 0.8269x | 102 | 0.00 / 0.00 | 37.47% | all8192 rows PASS, seeds1234/5678 |
| v068 | 2820.26 | 1691.87 | 0.5999x | 128 | 28.05 / 11.54 | 28.35% | prototype |
| v069 | 2043.84 | 1693.66 | 0.8287x | 102 | 0.00 / 0.00 | 37.65% | all8192 rows PASS, seeds1234/5678 |
| v071 | 2228.32 | 1691.90 | 0.7593x | 128 | 20.19 / 2.39 | 23.34% | prototype |
| v072 | 2218.21 | 1691.55 | 0.7626x | 128 | 11.01 / 1.86 | 23.40% | prototype |
| v073 | 2326.69 | 1691.74 | 0.7271x | 104 | 0.00 / 0.00 | 44.49% | prototype |
| v074 | 2371.71 | 1689.98 | 0.7126x | 93 | 0.00 / 0.00 | 24.41% | prototype |
| v075 | 1990.94 | 1691.81 | 0.8498x | 118 | 0.00 / 0.00 | 38.51% | all8192 rows PASS, seeds1234/5678 |
| v076 | 2146.50 | 1691.58 | 0.7881x | 168 | 0.00 / 0.00 | 35.46% | prototype |
| v077 | 1984.70 | 1691.84 | 0.8524x | 118 | 0.00 / 0.00 | 38.70% | all 8192 rows PASS, seed1234 |
| v078 | 1988.64 | 1693.76 | 0.8517x | 118 | 0.00 / 0.00 | 38.61% | prototype |
| v079 | 1992.90 | 1691.81 | 0.8489x | 118 | 0.00 / 0.00 | 38.53% | prototype |
| v080 | 2064.54 | 1691.90 | 0.8195x | 110 | 0.00 / 0.00 | 37.22% | prototype |
| v081 | 2021.57 | 1691.81 | 0.8369x | 109 | 0.00 / 0.00 | 38.02% | all8192 rows PASS, seeds1234/5678 |
| v082 | 2861.12 | 1691.68 | 0.5913x | 128 | 0.00 / 0.00 | 26.11% | prototype |
| v083 | 2027.90 | 1693.70 | 0.8352x | 128 | 0.00 / 0.00 | 37.62% | prototype |
| v084 | 1995.17 | 1692.06 | 0.8481x | 118 | 0.00 / 0.00 | 38.31% | all 8192 rows PASS, seed1234 |
| v085 | 1878.05 | 1691.97 | 0.9009x | 119 | 0.00 / 0.00 | 27.76% | same 9/6 all-row failures as TRT, seeds1234/5678 |
| v086 | 1870.05 | 1691.68 | 0.9046x | 119 | 0.00 / 0.00 | 27.89% | same 9/6 all-row failures as TRT, seeds1234/5678 |
| v087 | 1872.22 | 1693.86 | 0.9047x | 119 | 0.00 / 0.00 | 27.85% | bitwise v085 on all8192 rows, seed1234 |
| v088 | 2037.79 | 1690.85 | 0.8297x | 112 | 0.00 / 0.00 | 28.72% | bitwise v085 on all8192 rows, seed1234 |
| v089 | 1861.44 | 1691.74 | 0.9088x | 119 | 0.00 / 0.00 | 28.07% | bitwise v086 on full/short audited inputs |
| v090 | 1828.80 | 1691.49 | 0.9249x | 118 | 0.00 / 0.00 | 28.58% | bitwise v086, full2seeds/short/masks; FP8 limits retained |
| v091 | 1820.90 | 1691.81 | 0.9291x | 118 | 0.00 / 0.00 | 28.73% | bitwise v090, full2seeds/short/masks; FP8 limits retained |
| v092 | 2005.18 | 1691.81 | 0.8437x | 96 | 0.00 / 0.00 | 38.15% | bitwise v075 on all8192 rows, seed1234 |

Raw JSON records exact tensor shapes, seed, software versions, candidate SHA256 and unchanged benchmark SHA256. The baseline B0 used 20 warmups/100 repeats and measured 1860.70 µs warm / 1854.66 µs cold; use the paired baseline for each ratio because clocks vary. v090/v091 have an observed sustained warm advantage in the extended eager-event, Graph and rotating-order runs, at baseline-level FP8 precision; v091 is the current fast path. Short five-event tuning still favors TRT; use the matching execution regime. v086 is near parity warm and its initial cold advantage does not reproduce in its rotating-order audit.

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
| v063 | 512 | warm | 1913.23 | 1869.90 | 0.9774x |
| v063 | 512 | cold | 1900.56 | 1941.54 | 1.0216x |
| v065 | 512 | warm | 2279.70 | 1871.50 | 0.8209x |
| v065 | 512 | cold | 2269.34 | 1947.44 | 0.8582x |
| v067 | 512 | warm | 2134.14 | 1869.14 | 0.8758x |
| v067 | 512 | cold | 2107.38 | 1926.99 | 0.9144x |
| v075 | 512 | warm | 2097.22 | 1874.14 | 0.8936x |
| v075 | 512 | cold | 2080.51 | 1925.12 | 0.9253x |
| v085 | 512 | warm | 1888.18 | 1877.54 | 0.9944x |
| v085 | 512 | cold | 1887.87 | 1928.16 | 1.0213x |
| v086 | 512 | warm | 1873.54 | 1869.82 | 0.9980x |
| v086 | 512 | cold | 1880.13 | 1946.53 | 1.0353x |
| v087 | 512 | warm | 1889.58 | 1867.81 | 0.9885x |
| v087 | 512 | cold | 1882.14 | 1923.95 | 1.0222x |
| v088 | 512 | warm | 2037.95 | 1872.11 | 0.9186x |
| v088 | 512 | cold | 2045.87 | 1924.62 | 0.9407x |
| v089 | 512 | warm | 1882.21 | 1871.89 | 0.9945x |
| v089 | 512 | cold | 1871.87 | 1927.14 | 1.0295x |
| v090 | 512 | warm | 1853.31 | 1867.95 | 1.0079x |
| v090 | 512 | cold | 1839.36 | 1931.57 | 1.0501x |

## Extended eager CUDA-event runs

20 warmups and 100 repeats; the unchanged original timing function.

| Version | Checked rows | Cache | Candidate µs | Paired TRT µs | TRT/candidate |
|---|---:|---|---:|---:|---:|
| v090 | 512 | warm | 1859.63 | 1882.35 | 1.0122x |
| v090 | 512 | cold | 1845.01 | 1919.04 | 1.0401x |
| v091 | 512 | warm | 1852.51 | 1886.94 | 1.0186x |
| v091 | 512 | cold | 1851.06 | 1917.02 | 1.0356x |

## Same-process rotating-order audits

Ranges below are the minimum and maximum per-round medians or paired ratios, not confidence intervals.

| Audit | Candidate | Cache | Rounds | Candidate median range µs | Paired TRT/candidate range |
|---|---|---|---:|---:|---:|
| v086_v088_round_robin | cute-v086/native | warm | 3 | 1871.92–1872.16 | 0.9998–1.0000x |
| v086_v088_round_robin | cute-v086/native | cold | 3 | 1878.13–1878.30 | 0.9881–0.9891x |
| v086_v088_round_robin | cute-v088/native | warm | 3 | 2037.79–2037.89 | 0.9185–0.9186x |
| v086_v088_round_robin | cute-v088/native | cold | 3 | 2045.86–2045.90 | 0.9071–0.9080x |
| v086_v090_round_robin | cute-v086/native | warm | 3 | 1872.03–1872.06 | 0.9978–1.0032x |
| v086_v090_round_robin | cute-v086/native | cold | 3 | 1878.19–1879.44 | 0.9880–0.9901x |
| v086_v090_round_robin | cute-v090/native | warm | 3 | 1831.07–1831.63 | 1.0200–1.0257x |
| v086_v090_round_robin | cute-v090/native | cold | 3 | 1837.39–1838.98 | 1.0099–1.0113x |
| v090_v091_round_robin | cute-v090/native | warm | 3 | 1832.45–1833.10 | 1.0205–1.0254x |
| v090_v091_round_robin | cute-v090/native | cold | 3 | 1838.14–1839.01 | 1.0094–1.0124x |
| v090_v091_round_robin | cute-v091/native | warm | 3 | 1822.98–1824.83 | 1.0258–1.0300x |
| v090_v091_round_robin | cute-v091/native | cold | 3 | 1830.94–1830.98 | 1.0135–1.0168x |

See [ITERATIONS.md](ITERATIONS.md) for changes, failed hypotheses, correctness limits and source references.
