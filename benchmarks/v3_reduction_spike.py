"""Verify the Metal facts the v3 plan's reduction design rests on.

A: plan design — padded grid, PREDICATED exits, threadgroup memory,
   barrier, simdgroup-0 second stage.
B: challenger — exact grid, store-and-RETURN exits kept, simd_sum over
   the *active* lanes, first-active-lane store of per-simdgroup partials.
Both must reproduce a numpy masked per-chain sum; B additionally
tests whether early-returned lanes corrupt simd_sum.
"""
import numpy as np, mlx.core as mx

N, M = 5, 1000            # M deliberately not a multiple of 256 or 32
rng = np.random.default_rng(1)
x = rng.uniform(-1, 1, (N, M)).astype(np.float32)
active = x > 0.0          # "in transit" lanes; others early-exit with 0
ref = (x * active).sum(axis=1)

# ---- B: simd-only, early return allowed -------------------------------
srcB = """
    uint x = thread_position_in_grid.x;
    uint y = thread_position_in_grid.y;
    uint ng = ((uint)npts + 31) / 32;
    uint g  = x / 32;
    if (x >= (uint)npts) return;               // bounds: RETURN
    float v = xin[y * (uint)npts + x];
    if (v <= 0.0f) return;                     // early exit: RETURN
    float s = metal::simd_sum(v);
    if (metal::simd_is_first()) part[y * ng + g] = s;
"""
kB = mx.fast.metal_kernel(name="redB", input_names=["xin", "npts"],
                          output_names=["part"], source=srcB)
ng = (M + 31) // 32
partB = kB(inputs=[mx.array(x), M], output_shapes=[(N, ng)],
           output_dtypes=[mx.float32], init_value=0.0,
           grid=(M, N, 1), threadgroup=(256, 1, 1))[0]
sB = np.array(mx.sum(partB, axis=1))
print("B simd-only, returned lanes:   max|err| =",
      np.abs(sB - ref).max(), "(ref scale", np.abs(ref).max(), ")")

# ---- A: threadgroup memory + barrier, predicated ------------------------
srcA = """
    threadgroup float buf[8];
    uint x = thread_position_in_grid.x;
    uint y = thread_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    uint sg   = simdgroup_index_in_threadgroup;
    uint tg   = threadgroup_position_in_grid.x;
    uint ngr  = ((uint)npts + 255) / 256;
    float v = 0.0f;
    bool live = x < (uint)npts;
    if (live) { v = xin[y * (uint)npts + x]; if (v <= 0.0f) v = 0.0f; }
    float s = metal::simd_sum(v);
    if (lane == 0) buf[sg] = s;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (sg == 0) {
        float t = (lane < 8) ? buf[lane] : 0.0f;
        float tot = metal::simd_sum(t);
        if (lane == 0) part[y * ngr + tg] = tot;
    }
"""
kA = mx.fast.metal_kernel(name="redA", input_names=["xin", "npts"],
                          output_names=["part"], source=srcA)
ngr = (M + 255) // 256
mpad = ngr * 256
partA = kA(inputs=[mx.array(x), M], output_shapes=[(N, ngr)],
           output_dtypes=[mx.float32], init_value=0.0,
           grid=(mpad, N, 1), threadgroup=(256, 1, 1))[0]
sA = np.array(mx.sum(partA, axis=1))
print("A threadgroup+barrier padded: max|err| =", np.abs(sA - ref).max())

# ---- A without padding (exact grid): does a partial last group break? --
partA2 = kA(inputs=[mx.array(x), M], output_shapes=[(N, ngr)],
            output_dtypes=[mx.float32], init_value=0.0,
            grid=(M, N, 1), threadgroup=(256, 1, 1))[0]
sA2 = np.array(mx.sum(partA2, axis=1))
print("A with EXACT grid (nonuniform last group): max|err| =",
      np.abs(sA2 - ref).max())

# ---- determinism ----------------------------------------------------------
p2 = kB(inputs=[mx.array(x), M], output_shapes=[(N, ng)],
        output_dtypes=[mx.float32], init_value=0.0,
        grid=(M, N, 1), threadgroup=(256, 1, 1))[0]
print("B bitwise deterministic:", bool(mx.all(p2 == partB).item()))
