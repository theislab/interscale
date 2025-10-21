#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

from InterScale.module.global_modules.transformer_encoder import TransformerNodeEncoderHook

# Test the inheritance
print("Testing TransformerNodeEncoderHook inheritance...")

# Create a minimal instance
try:
    hook = TransformerNodeEncoderHook(
        max_seq_len=100,
        n_input=10,
        n_output=5,
        n_embed=16,
        pct_mask_nodes=0.1
    )
    
    print(f"Object type: {type(hook)}")
    print(f"Has masked_nodes attribute: {hasattr(hook, 'masked_nodes')}")
    if hasattr(hook, 'masked_nodes'):
        print(f"masked_nodes value: {hook.masked_nodes}")
    else:
        print("masked_nodes attribute not found!")
        
    # Check MRO (Method Resolution Order)
    print(f"MRO: {[cls.__name__ for cls in type(hook).__mro__]}")
    
    # Check if the attribute exists in any of the parent classes
    for cls in type(hook).__mro__:
        if hasattr(cls, '__dict__') and 'masked_nodes' in cls.__dict__:
            print(f"masked_nodes found in {cls.__name__}")
        else:
            print(f"masked_nodes NOT found in {cls.__name__}")
            
except Exception as e:
    print(f"Error creating TransformerNodeEncoderHook: {e}")
    import traceback
    traceback.print_exc()
