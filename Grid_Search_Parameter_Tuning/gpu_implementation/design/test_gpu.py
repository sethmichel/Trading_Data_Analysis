import cupy as cp
import cudf

print("Checking GPU...")
try:
    x = cp.array([1, 2, 3])
    print(f"CuPy Array created: {x}")
    print("CuPy Device:", cp.cuda.Device(0).compute_capability)

    df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    print("cuDF DataFrame created:\n", df)
    print("SUCCESS: GPU stack is working.")
except Exception as e:
    print("FAILURE:", e)