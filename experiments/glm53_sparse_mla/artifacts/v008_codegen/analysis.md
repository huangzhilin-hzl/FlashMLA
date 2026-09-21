# v008 generated-code inspection

Diagnostic build: b2, same kernel body; CuTeDSL 4.6.2, sm_103a,
`CUTE_DSL_KEEP=ptx,cubin,sass CUTE_DSL_LINEINFO=1`. These files are diagnostic
code generation, not full-target performance data. The binary remains local.

The PTX contains no local-memory instructions or explicit local arrays; the
SASS contains 90 static LDL/STL instructions. Thus these are backend register
allocation spills, not intentional local arrays. Static instruction counts are
not the dynamic sector counters from NCU.

39 static STL instructions are associated with line 104 source attribution;
31 LDLs with line 149, and 12 LDLs with line 136. PTX at line 149 shows an
unrolled sequence of `st.shared.b8` zero fills. Source attribution reflects
compiler transformations and should not be interpreted as a direct mapping
from every instruction to a single Python expression.

The masked-KV branch generates 18 x 16 scalar byte stores, with separate
swizzled destination addresses. Even on a chunk with no masked slots, hoisting
these addresses can enlarge the register live set and displace useful values.
The hypothesis will be tested by changing only that zero fill to a 16-byte
vector store in v010, compared with v008. Full-target NCU must confirm whether
local traffic disappears.
