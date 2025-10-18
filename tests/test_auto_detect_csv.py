"""
Test the stateless auto-detect CSV approach.
No global state - just method parameters and environment variables.
"""

import os
import tempfile
from pathlib import Path

from chronify import Store
from chronify.csv_utils import _should_use_auto_detect


def test_stateless_auto_detect_csv():
    """Test that auto-detection features work without global state."""
    
    # Create a test CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("timestamp,device,value\n")
        f.write("2020-01-01 00:00,A,100\n")
        f.write("2020-01-01 01:00,A,200\n")
        csv_path = f.name
    
    try:
        store = Store()
        
        # Test 1: Default behavior (no auto-detection)
        result = store.inspect_csv(csv_path)
        assert 'error' not in result
        print("✓ Default inspect_csv works")
        
        # Test 2: Explicitly enable auto-detection via parameter
        result = store.inspect_csv(csv_path, auto_detect=True)
        assert 'error' not in result
        print("✓ Auto-detect via parameter works")
        
        # Test 3: Environment variable (no persistence needed)
        os.environ["CHRONIFY_AUTO_DETECT_CSV"] = "true"
        result = store.inspect_csv(csv_path)  # Should detect from env
        assert 'error' not in result
        print("✓ Auto-detect via environment variable works")
        
        # Test 4: Parameter overrides environment
        result = store.inspect_csv(csv_path, auto_detect=False)  # Explicitly disabled
        assert 'error' not in result
        print("✓ Parameter override works")
        
        # Test 5: Utility function works correctly
        assert _should_use_auto_detect(None) == True  # From env
        assert _should_use_auto_detect(True) == True  # Explicit
        assert _should_use_auto_detect(False) == False  # Explicit override
        print("✓ Auto-detect detection works correctly")
        
        # Clean up environment
        del os.environ["CHRONIFY_AUTO_DETECT_CSV"]
        
        # Test 6: No environment, default to False
        assert _should_use_auto_detect(None) == False
        print("✓ Default fallback works")
        
        print("\n🎉 All stateless auto-detect CSV tests passed!")
        print("✅ No global state required")
        print("✅ Clean method parameters")
        print("✅ Environment variable support")
        print("✅ Parameter priority over environment")
        
    finally:
        # Cleanup
        Path(csv_path).unlink(missing_ok=True)
        if "CHRONIFY_AUTO_DETECT_CSV" in os.environ:
            del os.environ["CHRONIFY_AUTO_DETECT_CSV"]


if __name__ == "__main__":
    test_stateless_auto_detect_csv()