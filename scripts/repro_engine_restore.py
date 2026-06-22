import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_engine.native_mosaic as nm
inp = sys.argv[1]
out = sys.argv[2]
print("CUDA_GRAPH env:", os.environ.get("VRVT_CUDA_GRAPH"), "NATIVE_DCN:", os.environ.get("VRVT_NATIVE_DCN"))
print("reason:", nm.unavailable_reason())
ok = nm.restore_file(inp, out, max_clip_length=60, log_callback=lambda m: print(m))
print("restore_file returned:", ok)
