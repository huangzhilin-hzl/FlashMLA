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
| v093 | 1876.19 | 1691.84 | 0.9017x | 124 | 0.00 / 0.00 | 27.79% | bitwise v090 on full seed1234/mask; slower |
| v094 | 1816.99 | 1691.71 | 0.9311x | 120 | 0.00 / 0.00 | 28.74% | bitwise validated predecessor, full2seeds/short/masks; FP8 limits retained |
| v095 | 1816.64 | 1691.84 | 0.9313x | 118 | 0.00 / 0.00 | 28.81% | full seed1234 bitwise equivalence; see iteration log |
| v096 | 1761.38 | 1691.74 | 0.9605x | 118 | 0.00 / 0.00 | 29.31% | bitwise validated predecessor, full2seeds/short/masks; FP8 limits retained |
| v097 | 1749.06 | 1691.90 | 0.9673x | 120 | 0.00 / 0.00 | 29.51% | bitwise validated predecessor, full2seeds/short/masks; FP8 limits retained |
| v098 | 1924.90 | 1691.71 | 0.8789x | 118 | 0.00 / 0.00 | 39.41% | bitwise v075, full2seeds/short/masks; higher precision |
| v099 | 1968.35 | 1691.71 | 0.8595x | 80 | 70.13 / 32.55 | 26.18% | bitwise v097 full seed1234/mask; spills; slower |
| v100 | 1931.39 | 1691.58 | 0.8758x | 122 | 0.00 / 0.00 | 39.26% | bitwise v098 full seed1234/mask; no gain |
| v101 | 1808.42 | 1691.78 | 0.9355x | 168 | 10.49 / 3.71 | 28.50% | bitwise v097 full seed1234/mask; no gain |
| v102 | 1814.56 | 1692.10 | 0.9325x | 142 | 0.00 / 0.00 | 28.40% | bitwise v097 full seed1234/mask; no gain |
| v103 | 1757.22 | 1691.65 | 0.9627x | 168 | 0.00 / 0.00 | 29.45% | bitwise v097 full seed1234/mask; no gain |
| v104 | 1755.26 | 1691.68 | 0.9638x | 120 | 0.00 / 0.00 | 29.36% | bitwise v097 full seed1234/mask; no gain |
| v105 | 1693.86 | 1691.84 | 0.9988x | 85 | 0.00 / 0.00 | 30.53% | bitwise validated predecessor, full2seeds/short/masks; FP8 limits retained |
| v106 | 1851.68 | 1689.70 | 0.9125x | 100 | 0.00 / 0.00 | 41.23% | bitwise v098, full2seeds/short/masks; higher precision |
| v107 | 1685.60 | 1691.90 | 1.0037x | 85 | 0.00 / 0.00 | 30.74% | bitwise v105 full seed1234/mask; not promoted |
| v108 | 1680.42 | 1689.89 | 1.0056x | 85 | 0.00 / 0.00 | 30.75% | bitwise validated predecessor, full2seeds/short/masks; FP8 limits retained |
| v109 | 1796.42 | 1691.52 | 0.9416x | 85 | 0.00 / 0.00 | 28.70% | bitwise v105 full seed1234; slower overlap control |
| v110 | 1847.49 | 1692.26 | 0.9160x | 100 | 0.00 / 0.00 | 41.20% | bitwise v106 full seed1234/mask; no consistent gain |
| v111 | 1850.37 | 1693.79 | 0.9154x | 100 | 0.00 / 0.00 | 41.14% | bitwise v106 full seed1234/mask; no consistent gain |
| v112 | 1669.31 | 1693.76 | 1.0146x | 85 | 0.00 / 0.00 | 30.99% | same9/6 full-seed failures asTRT; earlier fast path |
| v113 | 2177.28 | 1691.65 | 0.7770x | 92 | 0.00 / 0.00 | 23.63% | smoke/eight-row and qualified sanitizers pass; slower; no full audit |
| v114 | 1828.93 | 1691.78 | 0.9250x | 122 | 0.00 / 0.00 | 41.67% | full2seeds/short/masks PASS; earlier higher precision |
| v115 | 1677.60 | 1691.84 | 1.0085x | 85 | 0.00 / 0.00 | 30.82% | bitwise predecessor full seed1234/mask; no short-run gain |
| v116 | 1835.04 | 1693.47 | 0.9229x | 122 | 0.00 / 0.00 | 41.45% | bitwise predecessor full seed1234/mask; no short-run gain |
| v117 | 2068.80 | 1691.58 | 0.8177x | 101 | 0.00 / 0.00 | 25.85% | smoke/eight-row and qualified sanitizers pass; slower; no full audit |
| v118 | 1898.53 | 1692.22 | 0.8913x | 112 | 0.00 / 0.00 | 27.15% | smoke/eight-row and qualified sanitizers pass; slower; no full audit |
| v119 | 1714.18 | 1691.65 | 0.9869x | 128 | 0.00 / 0.01 | 30.28% | bitwise v112 full seed1234/short; slower |
| v120 | 1831.14 | 1691.71 | 0.9239x | 122 | 0.00 / 0.00 | 41.55% | bitwise v114 full seed1234/mask; no short-run gain |
| v121 | 1784.00 | 1691.84 | 0.9483x | 128 | 71.67 / 41.19 | 28.93% | bitwise v112 full seed1234/mask; spills; slower |
| v123 | 1718.46 | 1691.74 | 0.9845x | 85 | 0.00 / 0.00 | 30.06% | bitwise v112 full seed1234/mask; slower |
| v125 | 1648.86 | 1690.75 | 1.0254x | 85 | 0.00 / 0.00 | 31.39% | bitwise v112 full2seeds/short/masks; earlier fast path |
| v126 | 2152.67 | 1691.78 | 0.7859x | 128 | 193.31 / 154.47 | 24.01% | bitwise v112 full seed1234/mask; more spills; slower |
| v127 | 1786.05 | 1689.98 | 0.9462x | 128 | 71.67 / 41.20 | 28.95% | bitwise v112 full seed1234/mask; spills unchanged; slower |
| v128 | 1802.37 | 1691.71 | 0.9386x | 122 | 0.00 / 0.00 | 42.19% | bitwise v114 full2seeds/short/masks; earlier higher precision |
| v129 | 1646.72 | 1693.63 | 1.0285x | 85 | 0.00 / 0.00 | 31.40% | bitwise v125 full2seeds/short/masks; cold gain, mixed warm; not promoted |
| v130 | 1646.94 | 1689.95 | 1.0261x | 85 | 0.00 / 0.00 | 31.45% | bitwise v125 full seed1234/mask; mixed cache-policy gain |
| v131 | 2037.98 | 1691.87 | 0.8302x | 84 | 0.00 / 0.00 | 27.33% | bitwise v125 full seed1234/mask; mandatory lookahead slower |
| v132 | 1661.15 | 1689.86 | 1.0173x | 84 | 0.00 / 0.00 | 31.13% | bitwise v125 full seed1234/mask; conditional lookahead, no net gain |
| v133 | 1802.05 | 1689.86 | 0.9377x | 122 | 0.00 / 0.00 | 42.27% | bitwise v128 full2seeds/short/masks; small cache-policy alternative |
| v134 | 1652.93 | 1691.84 | 1.0235x | 85 | 0.00 / 0.00 | 31.28% | bitwise v125 full seed1234/mask; no net scheduling gain |
| v136 | 2960.13 | 1691.74 | 0.5715x | 114 | 0.00 / 0.00 | 17.27% | guarded smoke/b512 and eight rows pass; slower; no full audit |
| v138 | 1639.46 | 1691.49 | 1.0317x | 123 | 0.00 / 0.00 | 31.57% | bitwise v125 full2seeds/short/masks; earlier fast path |
| v140 | 1822.75 | 1690.88 | 0.9277x | 123 | 0.00 / 0.00 | 41.79% | bitwise v128 full seed1234/mask; paired correction slower |
| v141 | 1636.54 | 1691.94 | 1.0338x | 123 | 0.00 / 0.00 | 31.57% | bitwise v138 full2seeds/short/masks; small cache-policy alternative |
| v142 | 1644.74 | 1690.78 | 1.0280x | 123 | 0.00 / 0.00 | 31.49% | bitwise v138 full seed1234/mask; paired epilogue slower |
| v143 | 1636.42 | 1691.78 | 1.0338x | 123 | 0.00 / 0.00 | 31.60% | bitwise v138 full2seeds/short/masks; small scheduling alternative |
| v144 | 1636.48 | 1689.89 | 1.0326x | 123 | 0.00 / 0.00 | 31.65% | bitwise v138 full2seeds/short/masks; small scheduling alternative |
| v145 | 1781.79 | 1691.90 | 0.9496x | 98 | 0.00 / 0.00 | 42.79% | bitwise v128 full2seeds/short/masks; earlier higher precision |
| v146 | 1628.10 | 1693.86 | 1.0404x | 123 | 0.00 / 0.00 | 31.78% | bitwise v138 full2seeds/short/masks; earlier fast path |
| v147 | 1690.85 | 1693.54 | 1.0016x | 126 | 0.00 / 0.00 | 30.64% | bitwise v138 full seed1234/mask; correction prefetch slower |
| v148 | 1749.12 | 1691.84 | 0.9673x | 98 | 0.00 / 0.00 | 43.65% | full2seeds/short/masks FP32 PASS; earlier higher precision |
| v149 | 1771.74 | 1691.42 | 0.9547x | 100 | 0.00 / 0.00 | 42.96% | bitwise v145 full2seeds/short/masks; superseded by v148 |
| v150 | 1624.38 | 1691.87 | 1.0415x | 123 | 0.00 / 0.00 | 31.84% | bitwise v146 full2seeds/short/masks; mixed scheduling gain |
| v151 | 1628.03 | 1691.78 | 1.0392x | 123 | 0.00 / 0.00 | 31.92% | bitwise v146 full seed1234/masks; no short-run gain |
| v152 | 1739.04 | 1691.84 | 0.9729x | 100 | 0.00 / 0.00 | 43.85% | bitwise v148 full2seeds/short/masks; warm gain, mixed cold |
| v153 | 1734.72 | 1691.90 | 0.9753x | 100 | 0.00 / 0.00 | 44.00% | bitwise v152 full2seeds/short/masks; earlier higher precision |
| v154 | 1743.71 | 1691.90 | 0.9703x | 100 | 0.00 / 0.00 | 43.82% | bitwise v152 full seed1234/masks; no short-run gain |
| v155 | 1804.42 | 1691.90 | 0.9376x | 100 | 0.00 / 0.00 | 42.19% | bitwise v152 full seed1234/masks; alternate PV tiles slower |
| v156 | 1753.18 | 1689.89 | 0.9639x | 100 | 0.00 / 0.00 | 43.62% | bitwise v152 full seed1234/masks; alternate PV tiles slower |
| v157 | 1732.61 | 1691.62 | 0.9763x | 100 | 0.00 / 0.00 | 44.08% | bitwise v152 full2seeds/short/masks; packed scaling alternative |
| v158 | 1627.23 | 1691.65 | 1.0396x | 123 | 0.00 / 0.00 | 31.80% | bitwise v146 full seed1234/masks; Q cache policy, no short gain |
| v159 | 1740.86 | 1689.98 | 0.9708x | 100 | 0.00 / 0.00 | 43.79% | bitwise v153 full seed1234/masks; Q cache policy slower |
| v160 | 1724.54 | 1689.70 | 0.9798x | 100 | 0.00 / 0.00 | 44.36% | bitwise v153 full2seeds/short/masks; earlier higher precision |
| v161 | 1634.46 | 1691.84 | 1.0351x | 123 | 0.00 / 0.00 | 31.63% | guarded memory/sync and full seed1234/mask bits pass; load/max slower |
| v162 | 1730.69 | 1691.94 | 0.9776x | 101 | 0.00 / 0.00 | 44.09% | guarded memory/sync and full seed1234/mask bits pass; load/max slower |
| v163 | 1855.87 | 1691.78 | 0.9116x | 109 | 0.00 / 0.00 | 27.74% | guarded checks and full seed1234/mask bits pass; mask fallback slower |
| v164 | 1957.82 | 1691.87 | 0.8642x | 107 | 0.00 / 0.00 | 38.72% | guarded checks and full seed1234/mask bits pass; mask fallback slower |
| v165 | 1672.03 | 1689.82 | 1.0106x | 123 | 0.00 / 0.00 | 30.95% | bitwise full seed1234/masks and qualified sanitizers pass; mask branch slower |
| v166 | 1988.77 | 1689.89 | 0.8497x | 105 | 0.00 / 0.00 | 38.05% | bitwise full seed1234/masks and qualified sanitizers pass; mask branch slower |
| v177 | 1700.00 | 1691.90 | 0.9952x | 113 | 0.00 / 0.00 | 30.38% | guarded checks/full seed1234/mask bits pass; recovers regression, slower than defaults |
| v178 | 1790.21 | 1693.70 | 0.9461x | 96 | 0.00 / 0.00 | 42.49% | guarded checks/full seed1234/mask bits pass; recovers regression, slower than defaults |
| v179 | 1631.49 | 1691.78 | 1.0370x | 123 | 0.00 / 0.00 | 31.74% | full seed1234/mask bits and sanitizers pass; role-only control has no short gain |
| v180 | 1728.96 | 1691.58 | 0.9784x | 101 | 0.00 / 0.00 | 44.17% | full seed1234/mask bits and sanitizers pass; role-only control has no short gain |
| v183 | 1550.50 | 1691.81 | 1.0911x | 123 | 0.00 / 0.00 | 33.52% | bitwise v146 full2seeds/short/masks; earlier fast path |
| v184 | 1693.89 | 1689.76 | 0.9976x | 122 | 0.00 / 0.00 | 45.19% | bitwise v160 full2seeds/short/masks; earlier higher precision |
| v185 | 1549.28 | 1691.74 | 1.0920x | 123 | 0.00 / 0.00 | 33.54% | full2seeds/short/masks bitwise; mixed small gain; validated alternative |
| v186 | 1689.79 | 1691.81 | 1.0012x | 122 | 0.00 / 0.00 | 45.24% | full2seeds/short/masks bitwise; mixed small gain; validated alternative |
| v190 | 1549.06 | 1691.94 | 1.0922x | 123 | 0.00 / 0.00 | 33.52% | bitwise v183 full2seeds/short/masks; earlier fast path |
| v191 | 1691.84 | 1691.71 | 0.9999x | 122 | 0.00 / 0.00 | 45.21% | bitwise v184 full2seeds/short/masks; earlier higher precision |
| v194 | 1560.77 | 1690.82 | 1.0833x | 128 | 0.00 / 0.00 | 33.29% | guarded checks/full seed1234/mask bits pass; register redistribution slower |
| v195 | 1673.41 | 1691.71 | 1.0109x | 128 | 0.00 / 0.00 | 45.82% | full2seeds/short/masks bitwise; register redistribution gain, superseded |
| v197 | 1662.30 | 1691.94 | 1.0178x | 128 | 0.00 / 0.00 | 46.15% | bitwise v184 full2seeds/short/masks; earlier higher precision |
| v198 | 1573.95 | 1689.92 | 1.0737x | 128 | 0.00 / 0.00 | 33.08% | guarded checks/full seed1234/mask bits pass; native x64 no gain |
| v199 | 1673.44 | 1691.78 | 1.0110x | 128 | 0.00 / 0.00 | 45.88% | guarded checks/full seed1234/mask bits pass; native x64 no gain |
| v200 | 1553.50 | 1691.62 | 1.0889x | 128 | 0.00 / 0.00 | 33.43% | guarded/full seed1234/mask checks pass; maximal role budget no default gain |
| v201 | 1665.22 | 1692.22 | 1.0162x | 128 | 0.00 / 0.00 | 46.04% | guarded/full seed1234/mask checks pass; maximal role budget no default gain |
| v202 | 1661.12 | 1691.46 | 1.0183x | 128 | 0.00 / 0.00 | 46.17% | full2seeds/short/masks bitwise; cache hint mixed warm gain; validated alternative |
| v205 | 1548.54 | 1691.87 | 1.0926x | 123 | 0.00 / 0.00 | 33.54% | full2seeds/short/masks bitwise; reciprocal handoff alternative; timing regime matters |
| v206 | 1661.12 | 1691.94 | 1.0186x | 128 | 0.00 / 0.00 | 46.18% | full2seeds/short/masks bitwise; reciprocal handoff alternative; timing regime matters |
| v207 | 1583.20 | 1691.68 | 1.0685x | 110 | 0.00 / 0.00 | 32.77% | guarded/full seed1234/mask checks pass; packed adjacent nodes slower |
| v208 | 1710.30 | 1691.90 | 0.9892x | 128 | 0.00 / 0.00 | 44.71% | guarded/full seed1234/mask checks pass; packed adjacent nodes slower |
| v209 | 1579.36 | 1691.78 | 1.0712x | 109 | 0.00 / 0.00 | 32.76% | guarded/full seed1234/mask checks pass; packed half-trees improve previous packing, still slower |
| v210 | 1689.86 | 1690.66 | 1.0005x | 128 | 0.00 / 0.00 | 45.39% | guarded/full seed1234/mask checks pass; packed half-trees improve previous packing, still slower |
| v213 | 1678.59 | 1691.52 | 1.0077x | 123 | 0.00 / 0.00 | 32.63% | guarded/full seed1234/mask checks pass; looped TMA coordinate reuse slower |
| v214 | 1751.30 | 1691.52 | 0.9659x | 128 | 0.00 / 0.00 | 45.58% | guarded/full seed1234/mask checks pass; looped TMA coordinate reuse slower |
| v215 | 1667.04 | 1691.84 | 1.0149x | 128 | 0.00 / 0.00 | 46.08% | full seed1234/masked FP32 and guarded checks pass; half residual conversion slower |
| v216 | 1673.38 | 1691.74 | 1.0110x | 128 | 0.00 / 0.00 | 45.83% | full seed1234/masked FP32 and guarded checks pass; half residual conversion slower |
| v217 | 1666.24 | 1693.98 | 1.0167x | 128 | 0.00 / 0.00 | 46.03% | guarded/full seed1234/mask equivalence pass; intermediate register budget no short gain |
| v219 | 1634.46 | 1693.60 | 1.0362x | 123 | 0.00 / 0.00 | 31.70% | guarded checks/full seed1234/mask parent equivalence; slower than defaults |
| v220 | 1736.93 | 1691.84 | 0.9740x | 128 | 0.00 / 0.00 | 44.13% | guarded checks/full seed1234/mask parent equivalence; slower than defaults |
| v221 | 1627.20 | 1689.73 | 1.0384x | 123 | 0.00 / 0.00 | 31.85% | guarded checks/full seed1234/mask parent equivalence; slower than defaults |
| v222 | 1720.67 | 1691.78 | 0.9832x | 128 | 0.00 / 0.00 | 44.64% | guarded checks/full seed1234/mask parent equivalence; slower than defaults |
| v223 | 1557.70 | 1691.49 | 1.0859x | 123 | 0.00 / 0.00 | 33.48% | guarded checks/full seed1234/mask parent equivalence; slower than defaults |
| v224 | 1674.18 | 1693.41 | 1.0115x | 128 | 0.00 / 0.00 | 45.88% | guarded checks/full seed1234/mask parent equivalence; slower than defaults |
| v225 | 1697.86 | 1691.74 | 0.9964x | 123 | 0.00 / 0.00 | 33.21% | guarded checks/full seed1234/mask parent equivalence; slower than defaults |
| v226 | 1750.30 | 1691.87 | 0.9666x | 128 | 0.00 / 0.00 | 46.05% | guarded checks/full seed1234/mask parent equivalence; slower than defaults |
| v227 | 1797.79 | 1689.95 | 0.9400x | 128 | 0.00 / 0.00 | 56.78% | guarded memory/sync pass;9 full seed1234 failures,66 mask failures; slower |
| v228 | 1798.11 | 1691.68 | 0.9408x | 128 | 0.00 / 0.00 | 56.82% | guarded checks/full seed1234/mask exact v227; no performance gain |
| v229 | 1896.70 | 1691.58 | 0.8919x | 128 | 0.00 / 0.00 | 79.07% | guarded checks/full seed1234 and masks FP32 PASS; slower; no other full audits |
| v230 | 1670.24 | 1691.81 | 1.0129x | 128 | 0.00 / 0.00 | 45.90% | guarded checks/full seed1234/mask parent equivalence; KV-tail P reuse slower |
| v231 | 1564.86 | 1691.65 | 1.0810x | 111 | 0.00 / 0.00 | 33.16% | guarded checks/full seed1234/mask parent equivalence; KV-tail P reuse slower |
| v232 | 1941.66 | 1691.71 | 0.8713x | 128 | 0.00 / 0.00 | 57.45% | guarded checks pass;9 full seed1234 failures,86 mask failures; single-buffer slower |
| v234 | 1671.42 | 1691.52 | 1.0120x | 128 | 0.00 / 0.00 | 44.95% | guarded/full seed1234 exact v228;9 full and66 mask failures; faster than normal control,slower than default |
| v235 | 1550.24 | 1690.14 | 1.0902x | 123 | 0.00 / 0.00 | 33.52% | guarded/full seed1234/short/mask exact v190; fast precision limits; producer unroll no gain |
| v240 | 1673.38 | 1691.90 | 1.0111x | 128 | 0.00 / 0.00 | 45.93% | guarded/full seed1234/short/mask exact v197; zero local traffic; unroll and role budgets slower |
| v241 | 1673.41 | 1691.97 | 1.0111x | 128 | 0.00 / 0.00 | 45.98% | guarded/full seed1234/short/mask exact v197; zero local traffic; unroll and role budgets slower |
| v242 | 1552.61 | 1691.87 | 1.0897x | 123 | 0.00 / 0.00 | 33.56% | guarded/full seed1234/short/mask parent equivalence; hardware KV release no gain |
| v243 | 1675.26 | 1691.78 | 1.0099x | 128 | 0.00 / 0.00 | 45.71% | guarded/full seed1234/short/mask parent equivalence; hardware KV release no gain |
| v246 | 1548.42 | 1690.78 | 1.0919x | 111 | 0.00 / 0.00 | 33.55% | guarded fixed/varlen sync and full/short/mask parent equivalence; valid single completion; no default gain |
| v247 | 1671.04 | 1691.81 | 1.0124x | 128 | 0.00 / 0.00 | 45.81% | guarded fixed/varlen sync and full/short/mask parent equivalence; valid single completion; no default gain |
| v248 | 1566.69 | 1691.84 | 1.0799x | 115 | 0.00 / 0.00 | 33.08% | guarded fixed/varlen sync and full/short/mask equivalence; delayed bitmap acquire no default gain |
| v249 | 1669.09 | 1691.65 | 1.0135x | 128 | 0.00 / 0.00 | 45.90% | guarded fixed/varlen sync and full/short/mask equivalence; delayed bitmap acquire no default gain |
| v250 | 1685.70 | 1690.88 | 1.0031x | 124 | 0.00 / 0.00 | 30.87% | guarded fixed/varlen and full/short/mask exact v242; early QK executes but slower |
| v251 | 1639.65 | 1691.78 | 1.0318x | 111 | 0.00 / 0.00 | 34.25% | guarded fixed/varlen and full/short/mask exact v250; delayed PV wait improves v250, slower than default |
| v252 | 1626.30 | 1691.65 | 1.0402x | 111 | 0.00 / 0.00 | 34.31% | guarded fixed/varlen and full/short/mask exact v251; single polling owner modestly faster, no default gain |
| v253 | 1652.96 | 1692.70 | 1.0240x | 118 | 0.00 / 0.00 | 33.46% | guarded fixed/varlen and full/short/mask exact v252; midpoint PV arbitration exercised but slower |
| v254 | 1624.16 | 1691.81 | 1.0417x | 123 | 0.00 / 0.00 | 30.63% | guarded fixed/varlen and full/short/mask exact v190; KV prefetch raises cache traffic and slows latency |
| v255 | 1546.34 | 1691.58 | 1.0939x | 123 | 0.00 / 0.00 | 33.54% | guarded fixed/varlen and full/short/mask exact v190; private index lookahead near default, no established gain |
| v256 | 1550.62 | 1691.68 | 1.0910x | 123 | 0.00 / 0.00 | 33.44% | guarded fixed/varlen and full/short/mask exact v255; index issue after publication no default gain |
| v257 | 1769.89 | 1689.76 | 0.9547x | 128 | 0.00 / 0.00 | 46.57% | guarded fixed/varlen and full/short/mask exact v243; strict dual-score overlap slower than default |
| v258 | 1548.42 | 1691.23 | 1.0922x | 123 | 0.00 / 0.00 | 33.53% | guarded fixed/varlen and full/short/mask exact v190; grouped QK descriptors no latency gain |
| v260 | 1549.15 | 1691.74 | 1.0920x | 123 | 0.00 / 0.00 | 33.49% | guarded fixed/varlen and full/short/mask exact v190; grouped QK descriptors no latency gain |
| v262 | 1675.20 | 1691.81 | 1.0099x | 128 | 0.00 / 0.00 | 45.83% | guarded fixed/varlen and full/short/mask exact v197; strict grouped QK after loop control slower |
| v263 | 1547.17 | 1689.92 | 1.0923x | 123 | 0.00 / 0.00 | 33.50% | guarded fixed/varlen and full/short/mask parent equivalence; index hint reduces total DRAM reads but no latency gain |
| v264 | 1667.30 | 1691.84 | 1.0147x | 128 | 0.00 / 0.00 | 46.01% | guarded fixed/varlen and full/short/mask parent equivalence; index hint reduces total DRAM reads but no latency gain |
| v265 | 1675.71 | 1691.74 | 1.0096x | 96 | 0.00 / 0.00 | 45.75% | bitwise parent full seed1234/short/masks; eight producers slower; not promoted |
| v266 | 1560.77 | 1691.78 | 1.0839x | 96 | 0.00 / 0.00 | 33.26% | bitwise parent full seed1234/short/masks; eight producers slower; not promoted |
| v271 | 1541.70 | 1691.97 | 1.0975x | 128 | 2.79 / 1.10 | 33.68% | bitwise v190 full2seeds/short/masks; mixed paired gain; not promoted |
| v272 | 1496.13 | 1693.70 | 1.1321x | 128 | 1.71 / 0.46 | 34.52% | bitwise v190 full2seeds/short/masks; earlier fast default; FP8 limits retained |
| v273 | 1608.03 | 1691.74 | 1.0521x | 128 | 1.71 / 0.46 | 47.33% | bitwise v197 full2seeds/short/masks; earlier strict default |
| v274 | 1493.25 | 1691.81 | 1.1330x | 128 | 1.75 / 0.49 | 34.51% | bitwise v272 full2seeds/short/masks; larger grids mixed; not promoted |
| v275 | 1496.06 | 1690.05 | 1.1297x | 128 | 1.83 / 0.54 | 34.55% | bitwise v272 full2seeds/short/masks; larger grids mixed; not promoted |
| v278 | 1490.21 | 1691.78 | 1.1353x | 128 | 0.89 / 0.44 | 34.54% | bitwise v272 full2seeds/short/masks; earlier fast default; FP8 limits retained |
| v279 | 1603.74 | 1691.68 | 1.0548x | 128 | 0.89 / 0.44 | 47.45% | bitwise v273 full2seeds/short/masks; mixed warm; not promoted |
| v280 | 1491.14 | 1691.81 | 1.1346x | 128 | 0.00 / 0.00 | 34.28% | bitwise fast parent full2seeds/short/masks; zero local traffic; mixed cold; not promoted |
| v281 | 1600.74 | 1691.84 | 1.0569x | 128 | 0.00 / 0.00 | 47.47% | bitwise strict parent full2seeds/short/masks; zero local traffic; earlier strict default |
| v284 | 1429.73 | 1691.74 | 1.1833x | 128 | 1.38 / 0.44 | 35.56% | bitwise v278 full2seeds/short/masks; current fast default; FP8 limits retained |
| v285 | 1537.15 | 1691.97 | 1.1007x | 128 | 0.00 / 0.00 | 48.97% | bitwise v281 full2seeds/short/masks; earlier strict default; zero local traffic |
| v286 | 1411.23 | 1689.92 | 1.1975x | 128 | 1.38 / 0.44 | 36.12% | bitwise v284 full2seeds/short/masks; rotated wins but standalone warm mixed; not promoted |
| v287 | 1530.08 | 1691.78 | 1.1057x | 128 | 0.00 / 0.00 | 49.11% | bitwise v285 full2seeds/short/masks; current strict default; zero local traffic |
| v288 | 1425.50 | 1691.65 | 1.1867x | 128 | 1.38 / 0.44 | 35.64% | guarded checks/full seed1234/short/masks exact v286; earlier Q slower in short tuning |

Raw JSON records exact tensor shapes, seed, software versions, candidate SHA256 and unchanged benchmark SHA256. The baseline B0 used 20 warmups/100 repeats and measured 1860.70 µs warm / 1854.66 µs cold; use the paired baseline for each ratio because clocks vary. v090/v091/v094/v096/v097/v105/v108/v112/v125 have an observed sustained warm advantage in the extended eager-event, Graph and rotating-order runs, at baseline-level FP8 precision; v284 is the current fast path. Short five-event v125 tuning has a small observed advantage; v105 is near parity. Use the matching execution regime and retained distributions. v086 is near parity warm and its initial cold advantage does not reproduce in its rotating-order audit.

## CUDA Graph validation runs

These are separate warm/cold runs with 20 warmups and 100 repeats; sampled row counts are shown explicitly.

| Version | Checked rows | Cache | Candidate µs | Paired TRT µs | TRT/candidate |
|---|---:|---|---:|---:|---:|
| v272 | 512 | warm | 1611.12 | 1872.00 | 1.1619x |
| v272 | 512 | cold | 1597.34 | 1918.86 | 1.2013x |
| v273 | 512 | warm | 1858.56 | 1866.38 | 1.0042x |
| v273 | 512 | cold | 1808.10 | 1916.19 | 1.0598x |
| v278 | 512 | warm | 1599.63 | 1869.22 | 1.1685x |
| v278 | 512 | cold | 1592.96 | 1919.14 | 1.2048x |
| v281 | 512 | warm | 1846.82 | 1865.79 | 1.0103x |
| v281 | 512 | cold | 1801.62 | 1915.81 | 1.0634x |
| v278 | 512 | warm | 1610.30 | 1868.42 | 1.1603x |
| v278 | 512 | cold | 1595.47 | 1911.71 | 1.1982x |
| v281 | 512 | warm | 1846.85 | 1869.76 | 1.0124x |
| v281 | 512 | cold | 1798.35 | 1916.98 | 1.0660x |
| v284 | 512 | warm | 1569.23 | 1865.81 | 1.1890x |
| v284 | 512 | cold | 1554.38 | 1916.86 | 1.2332x |
| v285 | 512 | warm | 1817.18 | 1858.59 | 1.0228x |
| v285 | 512 | cold | 1771.28 | 1911.94 | 1.0794x |
| v284 | 512 | warm | 1553.71 | 1870.16 | 1.2037x |
| v284 | 512 | cold | 1547.33 | 1904.64 | 1.2309x |
| v285 | 512 | warm | 1831.18 | 1873.76 | 1.0233x |
| v285 | 512 | cold | 1773.33 | 1902.66 | 1.0729x |
| v286 | 512 | warm | 1561.58 | 1865.74 | 1.1948x |
| v286 | 512 | cold | 1536.42 | 1914.91 | 1.2463x |
| v287 | 512 | warm | 1808.54 | 1815.55 | 1.0039x |
| v287 | 512 | cold | 1765.26 | 1904.67 | 1.0790x |
| v190 | 512 | warm | 1650.59 | 1867.89 | 1.1316x |
| v190 | 512 | cold | 1638.50 | 1915.82 | 1.1693x |
| v272 | 512 | warm | 1615.52 | 1871.07 | 1.1582x |
| v272 | 512 | cold | 1599.34 | 1914.91 | 1.1973x |
| v197 | 512 | warm | 1881.23 | 1860.37 | 0.9889x |
| v197 | 512 | cold | 1826.90 | 1894.53 | 1.0370x |
| v273 | 512 | warm | 1847.39 | 1865.87 | 1.0100x |
| v273 | 512 | cold | 1800.35 | 1900.59 | 1.0557x |
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
| v091 | 512 | warm | 1847.62 | 1863.94 | 1.0088x |
| v091 | 512 | cold | 1832.43 | 1926.14 | 1.0511x |
| v094 | 512 | warm | 1839.20 | 1865.82 | 1.0145x |
| v094 | 512 | cold | 1827.62 | 1927.14 | 1.0545x |
| v096 | 512 | warm | 1806.54 | 1880.22 | 1.0408x |
| v096 | 512 | cold | 1796.08 | 1920.86 | 1.0695x |
| v097 | 512 | warm | 1794.08 | 1869.82 | 1.0422x |
| v097 | 512 | cold | 1783.62 | 1918.70 | 1.0757x |
| v098 | 512 | warm | 2056.35 | 1871.82 | 0.9103x |
| v098 | 512 | cold | 2037.58 | 1918.86 | 0.9417x |
| v105 | 512 | warm | 1762.86 | 1867.81 | 1.0595x |
| v105 | 512 | cold | 1742.77 | 1916.77 | 1.0998x |
| v106 | 512 | warm | 1998.90 | 1867.82 | 0.9344x |
| v106 | 512 | cold | 1972.21 | 1916.93 | 0.9720x |
| v108 | 512 | warm | 1716.29 | 1872.02 | 1.0907x |
| v108 | 512 | cold | 1734.90 | 1933.12 | 1.1143x |
| v112 | 512 | warm | 1738.34 | 1869.89 | 1.0757x |
| v112 | 512 | cold | 1722.18 | 1919.95 | 1.1148x |
| v114 | 512 | warm | 1996.70 | 1867.87 | 0.9355x |
| v114 | 512 | cold | 1968.24 | 1918.93 | 0.9749x |
| v125 | 512 | warm | 1736.62 | 1871.82 | 1.0779x |
| v125 | 512 | cold | 1720.40 | 1914.78 | 1.1130x |
| v128 | 512 | warm | 1970.22 | 1871.52 | 0.9499x |
| v128 | 512 | cold | 1945.54 | 1916.35 | 0.9850x |
| v129 | 512 | warm | 1714.40 | 1865.87 | 1.0884x |
| v129 | 512 | cold | 1697.74 | 1915.04 | 1.1280x |
| v133 | 512 | warm | 1966.26 | 1867.84 | 0.9499x |
| v133 | 512 | cold | 1936.42 | 1917.06 | 0.9900x |
| v138 | 512 | warm | 1724.70 | 1869.02 | 1.0837x |
| v138 | 512 | cold | 1713.23 | 1920.93 | 1.1212x |
| v141 | 512 | warm | 1725.07 | 1871.66 | 1.0850x |
| v141 | 512 | cold | 1689.62 | 1925.84 | 1.1398x |
| v145 | 512 | warm | 1955.74 | 1867.89 | 0.9551x |
| v145 | 512 | cold | 1922.10 | 1915.04 | 0.9963x |
| v146 | 512 | warm | 1714.21 | 1867.89 | 1.0897x |
| v146 | 512 | cold | 1709.50 | 1919.25 | 1.1227x |
| v148 | 512 | warm | 1924.24 | 1867.82 | 0.9707x |
| v148 | 512 | cold | 1894.53 | 1894.32 | 0.9999x |
| v149 | 512 | warm | 1943.74 | 1865.90 | 0.9600x |
| v149 | 512 | cold | 1914.74 | 1899.58 | 0.9921x |
| v150 | 512 | warm | 1724.46 | 1867.81 | 1.0831x |
| v150 | 512 | cold | 1700.02 | 1914.94 | 1.1264x |
| v152 | 512 | warm | 1927.20 | 1876.02 | 0.9734x |
| v152 | 512 | cold | 1900.50 | 1917.01 | 1.0087x |
| v153 | 512 | warm | 1925.15 | 1873.78 | 0.9733x |
| v153 | 512 | cold | 1898.61 | 1917.02 | 1.0097x |
| v157 | 512 | warm | 1923.17 | 1869.97 | 0.9723x |
| v157 | 512 | cold | 1875.76 | 1900.58 | 1.0132x |
| v160 | 512 | warm | 1923.22 | 1873.89 | 0.9744x |
| v160 | 512 | cold | 1890.34 | 1914.67 | 1.0129x |

## Extended eager CUDA-event runs

20 warmups and 100 repeats; the unchanged original timing function.

| Version | Checked rows | Cache | Candidate µs | Paired TRT µs | TRT/candidate |
|---|---:|---|---:|---:|---:|
| v272 | 512 | warm | 1596.54 | 1890.26 | 1.1840x |
| v272 | 512 | cold | 1601.55 | 1916.99 | 1.1970x |
| v273 | 512 | warm | 1782.78 | 1880.67 | 1.0549x |
| v273 | 512 | cold | 1800.22 | 1917.28 | 1.0650x |
| v278 | 512 | warm | 1585.94 | 1872.00 | 1.1804x |
| v278 | 512 | cold | 1599.78 | 1915.17 | 1.1971x |
| v281 | 512 | warm | 1765.76 | 1879.97 | 1.0647x |
| v281 | 512 | cold | 1790.03 | 1916.42 | 1.0706x |
| v278 | 512 | warm | 1584.14 | 1878.06 | 1.1855x |
| v278 | 512 | cold | 1595.42 | 1916.77 | 1.2014x |
| v281 | 512 | warm | 1781.44 | 1879.18 | 1.0549x |
| v281 | 512 | cold | 1789.98 | 1914.91 | 1.0698x |
| v284 | 512 | warm | 1537.81 | 1882.03 | 1.2238x |
| v284 | 512 | cold | 1552.43 | 1921.31 | 1.2376x |
| v285 | 512 | warm | 1756.32 | 1869.89 | 1.0647x |
| v285 | 512 | cold | 1755.28 | 1916.96 | 1.0921x |
| v284 | 512 | warm | 1537.15 | 1880.06 | 1.2231x |
| v284 | 512 | cold | 1553.54 | 1917.02 | 1.2340x |
| v285 | 512 | warm | 1753.01 | 1873.76 | 1.0689x |
| v285 | 512 | cold | 1758.86 | 1918.99 | 1.0910x |
| v286 | 512 | warm | 1519.81 | 1876.53 | 1.2347x |
| v286 | 512 | cold | 1544.16 | 1917.15 | 1.2416x |
| v287 | 512 | warm | 1741.97 | 1869.74 | 1.0734x |
| v287 | 512 | cold | 1749.12 | 1918.94 | 1.0971x |
| v190 | 512 | warm | 1632.45 | 1878.14 | 1.1505x |
| v190 | 512 | cold | 1640.00 | 1911.01 | 1.1652x |
| v272 | 512 | warm | 1605.81 | 1878.03 | 1.1695x |
| v272 | 512 | cold | 1599.79 | 1916.83 | 1.1982x |
| v197 | 512 | warm | 1816.72 | 1876.00 | 1.0326x |
| v197 | 512 | cold | 1816.42 | 1901.23 | 1.0467x |
| v273 | 512 | warm | 1783.78 | 1871.60 | 1.0492x |
| v273 | 512 | cold | 1792.21 | 1915.46 | 1.0688x |
| v090 | 512 | warm | 1859.63 | 1882.35 | 1.0122x |
| v090 | 512 | cold | 1845.01 | 1919.04 | 1.0401x |
| v091 | 512 | warm | 1852.51 | 1886.94 | 1.0186x |
| v091 | 512 | cold | 1851.06 | 1917.02 | 1.0356x |
| v094 | 512 | warm | 1839.30 | 1874.00 | 1.0189x |
| v094 | 512 | cold | 1837.06 | 1923.28 | 1.0469x |
| v096 | 512 | warm | 1806.46 | 1886.18 | 1.0441x |
| v096 | 512 | cold | 1795.95 | 1916.88 | 1.0673x |
| v097 | 512 | warm | 1787.86 | 1883.94 | 1.0537x |
| v097 | 512 | cold | 1784.90 | 1929.81 | 1.0812x |
| v098 | 512 | warm | 2051.90 | 1887.10 | 0.9197x |
| v098 | 512 | cold | 2043.68 | 1916.66 | 0.9378x |
| v105 | 512 | warm | 1740.94 | 1876.64 | 1.0779x |
| v105 | 512 | cold | 1768.85 | 1913.94 | 1.0820x |
| v106 | 512 | warm | 1977.41 | 1874.05 | 0.9477x |
| v106 | 512 | cold | 1981.01 | 1918.05 | 0.9682x |
| v108 | 512 | warm | 1753.09 | 1876.54 | 1.0704x |
| v108 | 512 | cold | 1766.93 | 1915.55 | 1.0841x |
| v112 | 512 | warm | 1716.51 | 1888.85 | 1.1004x |
| v112 | 512 | cold | 1745.12 | 1915.10 | 1.0974x |
| v114 | 512 | warm | 1955.95 | 1882.72 | 0.9626x |
| v114 | 512 | cold | 1957.86 | 1915.01 | 0.9781x |
| v125 | 512 | warm | 1720.98 | 1890.26 | 1.0984x |
| v125 | 512 | cold | 1733.12 | 1911.52 | 1.1029x |
| v128 | 512 | warm | 1941.71 | 1880.22 | 0.9683x |
| v128 | 512 | cold | 1945.70 | 1916.98 | 0.9852x |
| v129 | 512 | warm | 1718.29 | 1877.76 | 1.0928x |
| v129 | 512 | cold | 1720.53 | 1911.38 | 1.1109x |
| v133 | 512 | warm | 1947.57 | 1878.99 | 0.9648x |
| v133 | 512 | cold | 1939.18 | 1918.99 | 0.9896x |
| v138 | 512 | warm | 1708.69 | 1880.22 | 1.1004x |
| v138 | 512 | cold | 1715.17 | 1914.93 | 1.1165x |
| v141 | 512 | warm | 1706.26 | 1884.32 | 1.1044x |
| v141 | 512 | cold | 1721.17 | 1917.01 | 1.1138x |
| v145 | 512 | warm | 1921.14 | 1880.14 | 0.9787x |
| v145 | 512 | cold | 1922.64 | 1914.94 | 0.9960x |
| v146 | 512 | warm | 1714.32 | 1881.14 | 1.0973x |
| v146 | 512 | cold | 1714.99 | 1914.91 | 1.1166x |
| v148 | 512 | warm | 1883.06 | 1880.13 | 0.9984x |
| v148 | 512 | cold | 1892.50 | 1902.13 | 1.0051x |
| v149 | 512 | warm | 1914.56 | 1869.23 | 0.9763x |
| v149 | 512 | cold | 1912.61 | 1916.59 | 1.0021x |
| v150 | 512 | warm | 1694.83 | 1866.38 | 1.1012x |
| v150 | 512 | cold | 1719.23 | 1921.20 | 1.1175x |
| v152 | 512 | warm | 1896.66 | 1888.08 | 0.9955x |
| v152 | 512 | cold | 1884.37 | 1916.90 | 1.0173x |
| v153 | 512 | warm | 1873.97 | 1880.13 | 1.0033x |
| v153 | 512 | cold | 1900.54 | 1914.83 | 1.0075x |
| v157 | 512 | warm | 1872.02 | 1884.03 | 1.0064x |
| v157 | 512 | cold | 1873.92 | 1914.85 | 1.0218x |
| v160 | 512 | warm | 1872.77 | 1884.06 | 1.0060x |
| v160 | 512 | cold | 1886.27 | 1914.90 | 1.0152x |

## Same-process rotating-order audits

Ranges below are the minimum and maximum per-round medians or paired ratios, not confidence intervals. Historical audits here include nvidia-smi queries between cases; the endpoint control found roughly209ms idle gaps that change the operating regime. Future helper runs default to queries off. See ITERATIONS.md for the controls and original standalone timing.

| Audit | Candidate | Cache | Rounds | Candidate median range µs | Paired TRT/candidate range |
|---|---|---|---:|---:|---:|
| fast_compute_release_round_robin | cute-v272/native | warm | 5 | 1601.74–1607.76 | 1.1647–1.2238x |
| fast_compute_release_round_robin | cute-v272/native | cold | 5 | 1597.70–1603.70 | 1.1957–1.2037x |
| fast_compute_release_round_robin | cute-v278/native | warm | 5 | 1593.52–1601.01 | 1.1751–1.2255x |
| fast_compute_release_round_robin | cute-v278/native | cold | 5 | 1596.91–1599.49 | 1.1971–1.2043x |
| fast_compute_release_round_robin | cute-v280/native | warm | 5 | 1593.98–1604.77 | 1.1669–1.2297x |
| fast_compute_release_round_robin | cute-v280/native | cold | 5 | 1579.10–1600.61 | 1.1985–1.2178x |
| fast_query_epilogue_overlap_round_robin | cute-v278/native | warm | 5 | 1596.96–1622.38 | 1.1526–1.2238x |
| fast_query_epilogue_overlap_round_robin | cute-v278/native | cold | 5 | 1593.22–1601.33 | 1.1946–1.2044x |
| fast_query_epilogue_overlap_round_robin | cute-v284/native | warm | 5 | 1562.69–1564.82 | 1.1966–1.2513x |
| fast_query_epilogue_overlap_round_robin | cute-v284/native | cold | 5 | 1546.19–1556.51 | 1.2319–1.2372x |
| fast_query_pv_overlap_round_robin | cute-v284/native | warm | 5 | 1562.90–1575.30 | 1.1885–1.2509x |
| fast_query_pv_overlap_round_robin | cute-v284/native | cold | 5 | 1548.19–1572.80 | 1.2195–1.2367x |
| fast_query_pv_overlap_round_robin | cute-v286/native | warm | 5 | 1545.60–1549.65 | 1.2113–1.2642x |
| fast_query_pv_overlap_round_robin | cute-v286/native | cold | 5 | 1537.71–1550.58 | 1.2369–1.2517x |
| persistent_grid_round_robin | cute-v272/native | warm | 5 | 1599.60–1613.65 | 1.1657–1.2240x |
| persistent_grid_round_robin | cute-v272/native | cold | 5 | 1599.58–1604.56 | 1.1940–1.2070x |
| persistent_grid_round_robin | cute-v274/native | warm | 5 | 1597.55–1607.70 | 1.1774–1.2224x |
| persistent_grid_round_robin | cute-v274/native | cold | 5 | 1599.46–1603.07 | 1.1961–1.2075x |
| persistent_grid_round_robin | cute-v275/native | warm | 5 | 1605.36–1609.84 | 1.1716–1.2195x |
| persistent_grid_round_robin | cute-v275/native | cold | 5 | 1599.57–1606.83 | 1.1963–1.2059x |
| query_reuse_round_robin | cute-v190/native | warm | 5 | 1628.48–1654.78 | 1.1562–1.1903x |
| query_reuse_round_robin | cute-v190/native | cold | 5 | 1634.22–1646.06 | 1.1650–1.1766x |
| query_reuse_round_robin | cute-v271/native | warm | 5 | 1638.98–1652.69 | 1.1488–1.1911x |
| query_reuse_round_robin | cute-v271/native | cold | 5 | 1634.38–1646.53 | 1.1654–1.1718x |
| query_reuse_round_robin | cute-v272/native | warm | 5 | 1603.47–1613.89 | 1.1712–1.2222x |
| query_reuse_round_robin | cute-v272/native | cold | 5 | 1601.57–1603.58 | 1.1952–1.1992x |
| strict_compute_release_round_robin | cute-v273/native | warm | 5 | 1825.06–1845.41 | 1.0154–1.0728x |
| strict_compute_release_round_robin | cute-v273/native | cold | 5 | 1795.97–1799.49 | 1.0650–1.0696x |
| strict_compute_release_round_robin | cute-v279/native | warm | 5 | 1822.58–1835.07 | 1.0267–1.0733x |
| strict_compute_release_round_robin | cute-v279/native | cold | 5 | 1791.71–1795.98 | 1.0675–1.0708x |
| strict_compute_release_round_robin | cute-v281/native | warm | 5 | 1818.58–1830.91 | 1.0269–1.0765x |
| strict_compute_release_round_robin | cute-v281/native | cold | 5 | 1788.03–1791.98 | 1.0697–1.0732x |
| strict_query_epilogue_overlap_round_robin | cute-v281/native | warm | 5 | 1818.70–1843.86 | 1.0142–1.0755x |
| strict_query_epilogue_overlap_round_robin | cute-v281/native | cold | 5 | 1787.87–1790.00 | 1.0682–1.0725x |
| strict_query_epilogue_overlap_round_robin | cute-v285/native | warm | 5 | 1809.95–1813.68 | 1.0311–1.0810x |
| strict_query_epilogue_overlap_round_robin | cute-v285/native | cold | 5 | 1761.26–1773.58 | 1.0787–1.0862x |
| strict_query_pv_overlap_round_robin | cute-v285/native | warm | 5 | 1804.37–1811.57 | 1.0254–1.0842x |
| strict_query_pv_overlap_round_robin | cute-v285/native | cold | 5 | 1768.02–1771.54 | 1.0773–1.0826x |
| strict_query_pv_overlap_round_robin | cute-v287/native | warm | 5 | 1796.16–1802.38 | 1.0336–1.0889x |
| strict_query_pv_overlap_round_robin | cute-v287/native | cold | 5 | 1759.33–1763.22 | 1.0826–1.0871x |
| strict_query_round_robin | cute-v197/native | warm | 5 | 1845.54–1867.89 | 1.0005–1.0487x |
| strict_query_round_robin | cute-v197/native | cold | 5 | 1828.86–1832.98 | 1.0401–1.0461x |
| strict_query_round_robin | cute-v273/native | warm | 5 | 1825.04–1832.48 | 1.0230–1.0624x |
| strict_query_round_robin | cute-v273/native | cold | 5 | 1792.22–1794.29 | 1.0604–1.0675x |
| v075_v098_round_robin | cute-v075/native | warm | 3 | 2043.94–2046.66 | 0.9143–0.9177x |
| v075_v098_round_robin | cute-v075/native | cold | 3 | 2047.81–2048.03 | 0.9091–0.9119x |
| v075_v098_round_robin | cute-v098/native | warm | 3 | 2005.14–2013.14 | 0.9320–0.9347x |
| v075_v098_round_robin | cute-v098/native | cold | 3 | 2009.02–2009.22 | 0.9266–0.9296x |
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
| v091_v094_v095_round_robin | cute-v091/native | warm | 4 | 1823.14–1825.09 | 1.0235–1.0247x |
| v091_v094_v095_round_robin | cute-v091/native | cold | 4 | 1830.82–1831.01 | 1.0134–1.0145x |
| v091_v094_v095_round_robin | cute-v094/native | warm | 4 | 1819.04–1819.97 | 1.0263–1.0278x |
| v091_v094_v095_round_robin | cute-v094/native | cold | 4 | 1825.94–1826.90 | 1.0157–1.0168x |
| v091_v094_v095_round_robin | cute-v095/native | warm | 4 | 1820.80–1821.02 | 1.0258–1.0269x |
| v091_v094_v095_round_robin | cute-v095/native | cold | 4 | 1826.66–1826.86 | 1.0157–1.0168x |
| v094_v096_round_robin | cute-v094/native | warm | 3 | 1820.78–1820.96 | 1.0269–1.0314x |
| v094_v096_round_robin | cute-v094/native | cold | 3 | 1826.93–1827.06 | 1.0166–1.0206x |
| v094_v096_round_robin | cute-v096/native | warm | 3 | 1769.98–1771.74 | 1.0554–1.0601x |
| v094_v096_round_robin | cute-v096/native | cold | 3 | 1767.50–1769.44 | 1.0498–1.0549x |
| v096_v097_round_robin | cute-v096/native | warm | 3 | 1771.58–1771.63 | 1.0544–1.0601x |
| v096_v097_round_robin | cute-v096/native | cold | 3 | 1767.82–1772.58 | 1.0479–1.0534x |
| v096_v097_round_robin | cute-v097/native | warm | 3 | 1757.28–1761.38 | 1.0616–1.0676x |
| v096_v097_round_robin | cute-v097/native | cold | 3 | 1755.33–1760.75 | 1.0549–1.0595x |
| v097_v105_round_robin | cute-v097/native | warm | 3 | 1757.30–1760.72 | 1.0524–1.0678x |
| v097_v105_round_robin | cute-v097/native | cold | 3 | 1759.20–1761.70 | 1.0556–1.0595x |
| v097_v105_round_robin | cute-v105/native | warm | 3 | 1726.42–1730.53 | 1.0714–1.0890x |
| v097_v105_round_robin | cute-v105/native | cold | 3 | 1712.43–1722.29 | 1.0822–1.0896x |
| v098_v106_round_robin | cute-v098/native | warm | 3 | 2005.17–2007.23 | 0.9315–0.9383x |
| v098_v106_round_robin | cute-v098/native | cold | 3 | 2008.46–2009.36 | 0.9244–0.9248x |
| v098_v106_round_robin | cute-v106/native | warm | 3 | 1962.75–1964.16 | 0.9510–0.9587x |
| v098_v106_round_robin | cute-v106/native | cold | 3 | 1955.70–1958.02 | 0.9486–0.9497x |
| v105_v107_v108_round_robin | cute-v105/native | warm | 4 | 1722.46–1730.75 | 1.0840–1.0881x |
| v105_v107_v108_round_robin | cute-v105/native | cold | 4 | 1714.32–1728.46 | 1.0770–1.0854x |
| v105_v107_v108_round_robin | cute-v107/native | warm | 4 | 1718.38–1724.56 | 1.0870–1.0940x |
| v105_v107_v108_round_robin | cute-v107/native | cold | 4 | 1714.13–1722.27 | 1.0833–1.0889x |
| v105_v107_v108_round_robin | cute-v108/native | warm | 4 | 1712.40–1720.51 | 1.0898–1.0978x |
| v105_v107_v108_round_robin | cute-v108/native | cold | 4 | 1708.00–1719.26 | 1.0816–1.0935x |
| v106_v110_v111_round_robin | cute-v106/native | warm | 4 | 1947.71–1964.19 | 0.9559–0.9624x |
| v106_v110_v111_round_robin | cute-v106/native | cold | 4 | 1954.83–1959.81 | 0.9478–0.9513x |
| v106_v110_v111_round_robin | cute-v110/native | warm | 4 | 1943.73–1960.03 | 0.9572–0.9593x |
| v106_v110_v111_round_robin | cute-v110/native | cold | 4 | 1948.42–1955.70 | 0.9509–0.9545x |
| v106_v110_v111_round_robin | cute-v111/native | warm | 4 | 1951.82–1964.10 | 0.9538–0.9584x |
| v106_v110_v111_round_robin | cute-v111/native | cold | 4 | 1953.86–1959.71 | 0.9479–0.9518x |
| v106_v114_round_robin | cute-v106/native | warm | 3 | 1951.84–1964.11 | 0.9529–0.9633x |
| v106_v114_round_robin | cute-v106/native | cold | 3 | 1953.76–1960.18 | 0.9483–0.9559x |
| v106_v114_round_robin | cute-v114/native | warm | 3 | 1939.41–1939.73 | 0.9641–0.9694x |
| v106_v114_round_robin | cute-v114/native | cold | 3 | 1927.15–1929.22 | 0.9638–0.9691x |
| v108_v112_round_robin | cute-v108/native | warm | 3 | 1712.34–1713.34 | 1.0922–1.0953x |
| v108_v112_round_robin | cute-v108/native | cold | 3 | 1708.10–1719.46 | 1.0803–1.0887x |
| v108_v112_round_robin | cute-v112/native | warm | 3 | 1701.89–1712.13 | 1.0957–1.1012x |
| v108_v112_round_robin | cute-v112/native | cold | 3 | 1697.63–1697.71 | 1.0941–1.0954x |
| v112_v125_round_robin | cute-v112/native | warm | 3 | 1700.98–1712.13 | 1.0945–1.1041x |
| v112_v125_round_robin | cute-v112/native | cold | 3 | 1695.87–1699.86 | 1.0932–1.0965x |
| v112_v125_round_robin | cute-v125/native | warm | 3 | 1688.99–1691.26 | 1.1078–1.1105x |
| v112_v125_round_robin | cute-v125/native | cold | 3 | 1687.54–1693.62 | 1.0972–1.1019x |
| v114_v128_round_robin | cute-v114/native | warm | 3 | 1939.54–1939.66 | 0.9638–0.9704x |
| v114_v128_round_robin | cute-v114/native | cold | 3 | 1927.20–1929.12 | 0.9629–0.9681x |
| v114_v128_round_robin | cute-v128/native | warm | 3 | 1922.99–1925.22 | 0.9721–0.9787x |
| v114_v128_round_robin | cute-v128/native | cold | 3 | 1916.85–1917.02 | 0.9689–0.9733x |
| v125_v129_repeat_round_robin | cute-v125/native | warm | 6 | 1681.02–1691.74 | 1.1054–1.1185x |
| v125_v129_repeat_round_robin | cute-v125/native | cold | 6 | 1683.38–1693.65 | 1.0964–1.1083x |
| v125_v129_repeat_round_robin | cute-v129/native | warm | 6 | 1681.39–1689.60 | 1.1097–1.1158x |
| v125_v129_repeat_round_robin | cute-v129/native | cold | 6 | 1675.39–1683.58 | 1.1031–1.1125x |
| v125_v129_v130_round_robin | cute-v125/native | warm | 4 | 1678.74–1690.18 | 1.1065–1.1138x |
| v125_v129_v130_round_robin | cute-v125/native | cold | 4 | 1680.96–1688.66 | 1.0973–1.1026x |
| v125_v129_v130_round_robin | cute-v129/native | warm | 4 | 1679.41–1688.54 | 1.1073–1.1147x |
| v125_v129_v130_round_robin | cute-v129/native | cold | 4 | 1675.33–1679.54 | 1.1035–1.1063x |
| v125_v129_v130_round_robin | cute-v130/native | warm | 4 | 1687.57–1689.86 | 1.1072–1.1105x |
| v125_v129_v130_round_robin | cute-v130/native | cold | 4 | 1681.30–1691.60 | 1.0946–1.1023x |
| v125_v129_v138_round_robin | cute-v125/native | warm | 4 | 1688.54–1690.40 | 1.1089–1.1134x |
| v125_v129_v138_round_robin | cute-v125/native | cold | 4 | 1685.52–1693.68 | 1.0968–1.1078x |
| v125_v129_v138_round_robin | cute-v129/native | warm | 4 | 1677.54–1689.39 | 1.1122–1.1186x |
| v125_v129_v138_round_robin | cute-v129/native | cold | 4 | 1681.36–1689.50 | 1.1001–1.1066x |
| v125_v129_v138_round_robin | cute-v138/native | warm | 4 | 1681.52–1682.54 | 1.1136–1.1191x |
| v125_v129_v138_round_robin | cute-v138/native | cold | 4 | 1675.25–1684.62 | 1.1027–1.1135x |
| v128_v133_round_robin | cute-v128/native | warm | 3 | 1923.10–1927.25 | 0.9712–0.9766x |
| v128_v133_round_robin | cute-v128/native | cold | 3 | 1916.94–1917.18 | 0.9700–0.9701x |
| v128_v133_round_robin | cute-v133/native | warm | 3 | 1913.10–1920.22 | 0.9738–0.9817x |
| v128_v133_round_robin | cute-v133/native | cold | 3 | 1898.53–1914.75 | 0.9712–0.9795x |
| v128_v133_v145_round_robin | cute-v128/native | warm | 4 | 1923.07–1925.25 | 0.9431–0.9764x |
| v128_v133_v145_round_robin | cute-v128/native | cold | 4 | 1916.77–1918.99 | 0.9712–0.9744x |
| v128_v133_v145_round_robin | cute-v133/native | warm | 4 | 1923.06–1925.78 | 0.9421–0.9762x |
| v128_v133_v145_round_robin | cute-v133/native | cold | 4 | 1912.77–1915.12 | 0.9733–0.9754x |
| v128_v133_v145_round_robin | cute-v145/native | warm | 4 | 1898.85–1900.75 | 0.9551–0.9898x |
| v128_v133_v145_round_robin | cute-v145/native | cold | 4 | 1896.43–1906.70 | 0.9775–0.9849x |
| v138_v141_round_robin | cute-v138/native | warm | 3 | 1681.54–1684.13 | 1.1109–1.1173x |
| v138_v141_round_robin | cute-v138/native | cold | 3 | 1675.38–1684.06 | 1.1029–1.1088x |
| v138_v141_round_robin | cute-v141/native | warm | 3 | 1679.46–1679.49 | 1.1122–1.1200x |
| v138_v141_round_robin | cute-v141/native | cold | 3 | 1672.27–1681.12 | 1.1050–1.1107x |
| v138_v143_v144_round_robin | cute-v138/native | warm | 4 | 1681.57–1682.51 | 1.1104–1.1179x |
| v138_v143_v144_round_robin | cute-v138/native | cold | 4 | 1677.20–1682.32 | 1.1061–1.1087x |
| v138_v143_v144_round_robin | cute-v143/native | warm | 4 | 1679.49–1683.46 | 1.1124–1.1179x |
| v138_v143_v144_round_robin | cute-v143/native | cold | 4 | 1672.99–1681.55 | 1.1058–1.1144x |
| v138_v143_v144_round_robin | cute-v144/native | warm | 4 | 1679.34–1681.65 | 1.1125–1.1194x |
| v138_v143_v144_round_robin | cute-v144/native | cold | 4 | 1673.12–1682.56 | 1.1080–1.1115x |
| v138_v144_v146_round_robin | cute-v138/native | warm | 4 | 1681.57–1682.62 | 1.1126–1.1177x |
| v138_v144_v146_round_robin | cute-v138/native | cold | 4 | 1679.47–1684.61 | 1.1046–1.1072x |
| v138_v144_v146_round_robin | cute-v144/native | warm | 4 | 1677.50–1679.50 | 1.1158–1.1204x |
| v138_v144_v146_round_robin | cute-v144/native | cold | 4 | 1672.82–1681.62 | 1.1071–1.1129x |
| v138_v144_v146_round_robin | cute-v146/native | warm | 4 | 1671.20–1673.34 | 1.1189–1.1239x |
| v138_v144_v146_round_robin | cute-v146/native | cold | 4 | 1666.53–1675.41 | 1.1098–1.1171x |
| v145_v148_v149_round_robin | cute-v145/native | warm | 4 | 1898.54–1898.77 | 0.9816–0.9861x |
| v145_v148_v149_round_robin | cute-v145/native | cold | 4 | 1891.30–1892.46 | 0.9805–0.9819x |
| v145_v148_v149_round_robin | cute-v148/native | warm | 4 | 1876.62–1880.00 | 0.9932–0.9970x |
| v145_v148_v149_round_robin | cute-v148/native | cold | 4 | 1869.84–1876.03 | 0.9892–0.9932x |
| v145_v148_v149_round_robin | cute-v149/native | warm | 4 | 1888.26–1900.70 | 0.9849–0.9912x |
| v145_v148_v149_round_robin | cute-v149/native | cold | 4 | 1882.27–1884.19 | 0.9848–0.9866x |
| v146_v150_round_robin | cute-v146/native | warm | 3 | 1671.42–1693.89 | 1.1088–1.1249x |
| v146_v150_round_robin | cute-v146/native | cold | 3 | 1665.14–1675.26 | 1.1101–1.1217x |
| v146_v150_round_robin | cute-v150/native | warm | 3 | 1669.10–1690.75 | 1.1131–1.1264x |
| v146_v150_round_robin | cute-v150/native | cold | 3 | 1662.98–1673.30 | 1.1113–1.1183x |
| v148_v152_round_robin | cute-v148/native | warm | 3 | 1880.06–1882.24 | 0.9935–0.9992x |
| v148_v152_round_robin | cute-v148/native | cold | 3 | 1870.64–1871.87 | 0.9931–0.9957x |
| v148_v152_round_robin | cute-v152/native | warm | 3 | 1872.91–1873.87 | 0.9973–1.0036x |
| v148_v152_round_robin | cute-v152/native | cold | 3 | 1859.79–1872.13 | 0.9923–1.0021x |
| v148_v152_v153_round_robin | cute-v148/native | warm | 4 | 1878.69–1882.10 | 0.9953–0.9998x |
| v148_v152_v153_round_robin | cute-v148/native | cold | 4 | 1869.70–1873.94 | 0.9957–0.9981x |
| v148_v152_v153_round_robin | cute-v152/native | warm | 4 | 1871.86–1876.05 | 0.9989–1.0039x |
| v148_v152_v153_round_robin | cute-v152/native | cold | 4 | 1859.98–1873.84 | 0.9952–1.0041x |
| v148_v152_v153_round_robin | cute-v153/native | warm | 4 | 1869.70–1871.78 | 1.0001–1.0055x |
| v148_v152_v153_round_robin | cute-v153/native | cold | 4 | 1865.89–1867.68 | 0.9973–1.0001x |
| v152_v153_v157_round_robin | cute-v152/native | warm | 4 | 1868.00–1873.82 | 0.9982–1.0020x |
| v152_v153_v157_round_robin | cute-v152/native | cold | 4 | 1859.65–1861.66 | 0.9972–0.9998x |
| v152_v153_v157_round_robin | cute-v153/native | warm | 4 | 1865.52–1867.74 | 1.0003–1.0040x |
| v152_v153_v157_round_robin | cute-v153/native | cold | 4 | 1853.98–1863.70 | 0.9983–1.0019x |
| v152_v153_v157_round_robin | cute-v157/native | warm | 4 | 1863.65–1865.65 | 1.0015–1.0051x |
| v152_v153_v157_round_robin | cute-v157/native | cold | 4 | 1851.25–1863.10 | 0.9984–1.0033x |
| v153_v157_v160_round_robin | cute-v153/native | warm | 4 | 1867.79–1871.90 | 0.9992–1.0043x |
| v153_v157_v160_round_robin | cute-v153/native | cold | 4 | 1857.74–1868.05 | 0.9958–1.0021x |
| v153_v157_v160_round_robin | cute-v157/native | warm | 4 | 1865.66–1867.92 | 1.0003–1.0055x |
| v153_v157_v160_round_robin | cute-v157/native | cold | 4 | 1863.63–1865.49 | 0.9980–1.0001x |
| v153_v157_v160_round_robin | cute-v160/native | warm | 4 | 1861.50–1861.78 | 1.0024–1.0088x |
| v153_v157_v160_round_robin | cute-v160/native | cold | 4 | 1857.86–1861.58 | 1.0001–1.0033x |

See [ITERATIONS.md](ITERATIONS.md) for changes, failed hypotheses, correctness limits and source references.
