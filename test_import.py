#!/usr/bin/env python3
"""Test script to check if circular imports are resolved."""

try:
    print("Testing InterScale imports...")
    
    # Test basic imports
    print("1. Testing basic module imports...")
    from InterScale import config
    print("   ✓ config imported successfully")
    
    from InterScale import module
    print("   ✓ module imported successfully")
    
    from InterScale import tl
    print("   ✓ tl imported successfully")
    
    from InterScale import model
    print("   ✓ model imported successfully")
    
    print("\n2. Testing specific class imports...")
    from InterScale.model.base._base_model import BaseModelClass
    print("   ✓ BaseModelClass imported successfully")
    
    from InterScale.module.base._base_module import BaseModuleClass
    print("   ✓ BaseModuleClass imported successfully")
    
    from InterScale.train._trainingplans import TrainingPlan
    print("   ✓ TrainingPlan imported successfully")
    
    print("\n3. Testing model classes...")
    from InterScale.model import LocalModel, GlobalModel, CombinedModel
    print("   ✓ All model classes imported successfully")
    
    print("\n🎉 All imports successful! Circular import issue resolved.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
