import triton.language as tl
try:
    print(f"tl.dot_scaled available: {hasattr(tl, 'dot_scaled')}")
except Exception as e:
    print(f"Error checking tl.dot_scaled: {e}")
