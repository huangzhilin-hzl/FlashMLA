"""Compare normal-M64 A collector reuse within versus across datapath halves.

Exact integer FP8 operands and CTA-dependent row permutations distinguish
numerical failures from rounding or accidentally identical input matrices.
No MLA performance claim; guarding is optional to diagnose instrumentation effects.
"""
import argparse,hashlib,json,os
from pathlib import Path
import sys

if __name__=='__main__' and '--help' not in sys.argv:
    pre=argparse.ArgumentParser(add_help=False)
    pre.add_argument('--output-dir',type=Path)
    opts,_=pre.parse_known_args()
    if opts.output_dir is not None:
        target=opts.output_dir.resolve();target.mkdir(parents=True,exist_ok=True)
        os.chdir(target);sys.argv+=['--output-dir',str(target)]

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as bw
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.cute.runtime import from_dlpack
from kernel_v233 import mma_normal,tmem_before_sync,tmem_after_sync

class CollectorProbe:
    def __init__(self,mode):self.mode=mode

    @cute.jit
    def __call__(self,a:cute.Tensor,b:cute.Tensor,out:cute.Tensor,stream:cuda.CUstream):
        fp8=cutlass.Float8E4M3FN
        mma=bw.make_trivial_tiled_mma(fp8,OperandMajorMode.K,OperandMajorMode.K,
            cutlass.Float32,tcgen05.CtaGroup.ONE,(64,128))
        atom=tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW32,fp8)
        al=cute.tile_to_shape(atom,(64,32),order=(0,1))
        bl=cute.tile_to_shape(atom,(512,32),order=(0,1))
        self.kernel(a,b,out,mma,al,bl).launch(grid=(out.shape[0],1,1),block=(128,1,1),stream=stream)

    @cute.kernel
    def kernel(self,a:cute.Tensor,b:cute.Tensor,out:cute.Tensor,mma:cute.TiledMma,
               al:cute.ComposedLayout,bl:cute.ComposedLayout):
        tid,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx()
        warp=cute.arch.warp_idx();alloc=utils.SmemAllocator()
        sa=alloc.allocate_tensor(cutlass.Float8E4M3FN,al.outer,byte_alignment=128,swizzle=al.inner)
        sb=alloc.allocate_tensor(cutlass.Float8E4M3FN,bl.outer,byte_alignment=128,swizzle=bl.inner)
        done=alloc.allocate_array(cutlass.Int64,1)
        holding=alloc.allocate_array(cutlass.Int32,1)
        if tid==0:cute.arch.mbarrier_init(done,1)
        cute.arch.mbarrier_init_fence()
        for i in cutlass.range(tid,64*32,128):sa[i//32,i%32]=a[(i//32+block)%64,i%32]
        for i in cutlass.range(tid,512*32,128):sb[i//32,i%32]=b[i//32,i%32]
        if warp==0:
            cute.arch.alloc_tmem(256,holding)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.barrier();cute.arch.fence_view_async_shared()
        tp=cute.arch.retrieve_tmem_ptr(cutlass.Float32,alignment=16,ptr_to_buffer_holding_addr=holding)
        if warp==0:
            with cute.arch.elect_one():
                if cutlass.const_expr(self.mode=='same'):
                    for plane in cutlass.range(2,unroll_full=True):
                        mma_normal(tp+plane*(16<<16),sa,
                            cute.local_tile(sb,(128,32),(plane*2,0)),128,False,False,'.collector::a::fill')
                        mma_normal(tp+plane*(16<<16)+128,sa,
                            cute.local_tile(sb,(128,32),(plane*2+1,0)),128,False,False,'.collector::a::lastuse')
                else:
                    for chunk in cutlass.range(2,unroll_full=True):
                        if cutlass.const_expr(self.mode=='cross'):
                            mma_normal(tp+chunk*128,sa,
                                cute.local_tile(sb,(128,32),(chunk,0)),128,False,False,'.collector::a::fill')
                            mma_normal(tp+(16<<16)+chunk*128,sa,
                                cute.local_tile(sb,(128,32),(chunk+2,0)),128,False,False,'.collector::a::lastuse')
                        else:
                            mma_normal(tp+chunk*128,sa,
                                cute.local_tile(sb,(128,32),(chunk,0)),128,False,False)
                            mma_normal(tp+(16<<16)+chunk*128,sa,
                                cute.local_tile(sb,(128,32),(chunk+2,0)),128,False,False)
                tcgen05.commit(done)
        cute.arch.mbarrier_wait(done,0);tmem_after_sync()
        cf=mma.make_fragment_C(mma.partition_shape_C((64,128)))
        first=cf[((None,None),0,0)]
        layout=cute.make_layout((first.shape[0],128),stride=(first.stride[0],first.stride[1]))
        base_tile=cute.make_tensor(tp,layout)
        load=tcgen05.make_tmem_copy(cute.make_copy_atom(
            tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(64)),cutlass.Float32),base_tile)
        lane=load.get_slice(tid)
        coords=lane.partition_D(cute.make_identity_tensor((64,128)))
        values=cute.make_fragment_like(coords,cutlass.Float32)
        for plane in cutlass.range(2,unroll_full=True):
            for chunk in cutlass.range(2,unroll_full=True):
                tile=cute.make_tensor(tp+plane*(16<<16)+chunk*128,layout)
                cute.copy(load,lane.partition_S(tile),values)
                cute.arch.fence_view_async_tmem_load()
                for j in cutlass.range(cute.size(values),unroll_full=True):
                    h,c=coords[j];out[block,h,plane*256+chunk*128+c]=values[j]
        tmem_before_sync();cute.arch.barrier();tmem_after_sync()
        if warp==0:cute.arch.dealloc_tmem(tp,256)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blocks',type=int,default=1)
    p.add_argument('--mode',choices=['discard','same','cross'],required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--guardrails',action='store_true')
    p.add_argument('--compile-only',action='store_true')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    assert os.getenv('CUDA_VISIBLE_DEVICES')=='GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2'
    torch.manual_seed(9171);torch.backends.cuda.matmul.allow_tf32=False
    a=torch.randint(-3,4,(64,32),device='cuda').to(torch.float8_e4m3fn)
    b=torch.randint(-3,4,(512,32),device='cuda').to(torch.float8_e4m3fn)
    out=torch.empty((args.blocks,64,512),dtype=torch.float32,device='cuda')
    tensors=[from_dlpack(t,assumed_align=16) for t in [a,b,out]]
    stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    options='--keep-cubin --keep-ptx'
    if args.guardrails:options+=' --ptxas-options=-g-tmem-access-check'
    fn=cute.compile(CollectorProbe(args.mode),*tensors,stream,options=options)
    report={'purpose':__doc__,'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'helper_sha256':hashlib.sha256(Path(sys.modules['kernel_v233'].__file__).read_bytes()).hexdigest(),
            'records':[]}
    if not args.compile_only:
        ref0=a.float()@b.float().T
        index=(torch.arange(64,device='cuda')[None,:]+torch.arange(args.blocks,device='cuda')[:,None])%64
        ref=ref0[index]
        for i in range(args.repeats):
            fn(*tensors,stream);torch.cuda.synchronize()
            bad=out!=ref
            rec={'repeat':i,'elements':out.numel(),'mismatches':int(bad.sum().item()),
                 'plane_mismatches':[int(bad[:,:,j*256:(j+1)*256].sum().item()) for j in range(2)],
                 'max_abs':float((out-ref).abs().max().item())}
            report['records'].append(rec);print(json.dumps(rec),flush=True)
    (args.output_dir/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PROBE RECORD COMPLETE',flush=True)
