"""
Quick test of walk-forward engine basic functionality
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analysis.walk_forward.walk_forward_engine import WalkForwardEngine

print("=" * 80)
print("Walk-Forward Engine Quick Test")
print("=" * 80)

# Initialize engine
print("\n1. Initializing engine...")
try:
    engine = WalkForwardEngine(verbose=False)
    print("   OK: Engine initialized")
    print(f"   Price data shape: {engine.price_df.shape}")
    print(f"   Return data shape: {engine.ret_df.shape}")
    print(f"   Factor files: {len(engine.factor_files)}")
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test window generation
print("\n2. Testing window generation...")
try:
    walks = engine._generate_walk_windows()
    print(f"   OK: Generated {len(walks)} walks")
    if len(walks) > 0:
        train_start, train_end, test_start, test_end = walks[0]
        print(f"   First walk:")
        print(f"     Train: {train_start.date()} to {train_end.date()} ({(train_end - train_start).days} days)")
        print(f"     Test:  {test_start.date()} to {test_end.date()} ({(test_end - test_start).days} days)")
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test factor processing
print("\n3. Testing factor processing...")
try:
    if len(walks) > 0:
        train_start, train_end, test_start, test_end = walks[0]
        processed = engine._process_factors_rolling(train_end)
        print(f"   OK: Processed {len(processed)} factors")
        if processed:
            first_factor = list(processed.keys())[0]
            df = processed[first_factor]
            print(f"   Sample factor: {first_factor}")
            print(f"     Shape: {df.shape}")
            print(f"     Date range: {df.index.min().date()} to {df.index.max().date()}")
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("Quick test completed successfully!")
print("=" * 80)
print("\nYou can now run the full walk-forward validation:")
print("  python analysis/walk_forward/run_walk_forward.py")
print()
