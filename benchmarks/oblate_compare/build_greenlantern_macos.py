"""Build pocky + GreenLantern against Apple's OpenCL 1.2 (macOS).

Run from benchmarks/oblate_compare after cloning both into vendor/ (see
README.md). Idempotent. Compatibility shims only -- no algorithmic change:

  pocky        <OpenCL/opencl.h> on Apple; clCreateCommandQueue (1.2) for
               clCreateCommandQueueWithProperties (2.0); -framework OpenCL
  greenlantern -framework OpenCL; its embedded-kernel header (upstream
               generates it only in sdist) generated here with orbit.cl
               first, work_group_barrier -> barrier, and
               work_group_reduce_add (2.0) -> a local-memory tree reduction
               over the flattened 2-D work group (same sums, other order)

Then:  .venv/bin/pip install --no-build-isolation -e vendor/pocky
       .venv/bin/pip install --no-build-isolation -e vendor/greenlantern
"""
import glob
import os
import re

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")


def _sub(path, old, new):
    s = open(path).read()
    if new in s:
        return
    assert old in s, (path, old[:60])
    open(path, "w").write(s.replace(old, new, 1))


def patch_pocky():
    d = os.path.join(VENDOR, "pocky")
    _sub(os.path.join(d, "src/pocky/ext/include/pocky.h"), "#include <CL/opencl.h>",
         "#ifdef __APPLE__\n#include <OpenCL/opencl.h>\n#else\n#include <CL/opencl.h>\n#endif")
    _sub(os.path.join(d, "src/pocky/ext/pocky_utils.c"),
         "queue = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);",
         "#ifdef __APPLE__\n        queue = clCreateCommandQueue(ctx, dev, 0, &err);   /* OpenCL 1.2 */\n"
         "#else\n        queue = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);\n#endif")
    p = os.path.join(d, "setup.py")
    _sub(p, "import os\nimport numpy", "import os\nimport sys\nimport numpy")
    _sub(p, "Extension(name='pocky.ext', sources=source_files, libraries=['OpenCL'],",
         "Extension(name='pocky.ext', sources=source_files,\n"
         "        libraries=[] if sys.platform == 'darwin' else ['OpenCL'],\n"
         "        extra_link_args=['-framework', 'OpenCL'] if sys.platform == 'darwin' else [],")


def patch_greenlantern():
    d = os.path.join(VENDOR, "greenlantern")
    p = os.path.join(d, "setup.py")
    _sub(p, "import os\nimport glob", "import os\nimport sys\nimport glob")
    _sub(p, "        libraries=['OpenCL'], depends=header_files)",
         "        libraries=[] if sys.platform == 'darwin' else ['OpenCL'],\n"
         "        extra_link_args=['-framework', 'OpenCL'] if sys.platform == 'darwin' else [],\n"
         "        depends=header_files)")
    os.chdir(d)
    _generate_kernels()


KDIR, HDIR = "src/greenlantern/ext/kernels", "src/greenlantern/ext/include"
SHIM = r"""
#if !defined(__OPENCL_C_VERSION__) || __OPENCL_C_VERSION__ < 200
#define work_group_barrier(flags) barrier(flags)
#endif
#define WG_SCRATCH (1024)
float wg_reduce_add(float v, local float *sc)
{
    int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int n = get_local_size(0) * get_local_size(1);
    sc[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    int p2 = 1;
    while (p2 < n) p2 <<= 1;
    for (int s = p2 >> 1; s > 0; s >>= 1)
    {
        if (lid < s && lid + s < n) sc[lid] += sc[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float r = sc[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}
"""

def patch(src):
    out, pos = [], 0
    for m in re.finditer(r"kernel void \w+\([^)]*\)\s*\{", src):
        out.append(src[pos:m.end()])
        # body extent: brace matching
        depth, i = 1, m.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(src[i], 0)
            i += 1
        body = src[m.end():i]
        if "work_group_reduce_add" in body:
            body = ("\n    local float wg_scratch[WG_SCRATCH];" +
                    re.sub(r"work_group_reduce_add\(([^;]*?)\);",
                           r"wg_reduce_add(\1, wg_scratch);", body))
        out.append(body)
        pos = i
    out.append(src[pos:])
    return "".join(out)

def _generate_kernels():
    order = ["orbit", "ellipsoid", "ellipsoid_dual"]
    paths = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(os.path.join(KDIR, "*.cl"))}
    assert sorted(paths) == sorted(order), paths
    with open(os.path.join(HDIR, "greenlantern_kernels.h"), "w") as f:
        names = []
        for k, name in enumerate(order):
            content = patch(open(paths[name]).read())
            assert "work_group_reduce_add" not in content, name
            if k == 0:
                content = SHIM + content
            f.write(f'const char kernel_{name}[] = \n"' + '\\n"\n"'.join(
                content.replace("\\", "\\\\").replace('"', '\\"').split("\n"))
                + '\\n";\n\n')
            names.append(f"kernel_{name}")
        f.write(f"const cl_uint num_kernel_frags = {len(names)};\n"
                f"const char *kernel_frags[] = {{ {', '.join(names)} }};\n")
    print("generated", order)


if __name__ == "__main__":
    patch_pocky()
    patch_greenlantern()
