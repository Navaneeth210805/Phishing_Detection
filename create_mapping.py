#!/usr/bin/env python3
import json

# Read mapping from first training entry
with open("dom_brand_stage1_results.jsonl", 'r') as f:
    for line in f:
        data = json.loads(line)
        if data.get('cmd') == 'mc-train' and data.get('algorithm') == 'random_forest':
            # Extract mapping info
            class_hist = data.get('class_hist_raw', {})
            top_targets = list(class_hist.keys())[:-1]  # Remove 'unknown_agg'
            
            mapping = {
                "source_split": "train",
                "top_k": len(top_targets),
                "top_targets": top_targets,
                "unknown_class_name": "unknown_agg",
                "unknown_class_id": len(top_targets),
            }
            
            # Save mapping
            with open("dom_stage1_mapping.json", 'w') as mf:
                json.dump(mapping, mf, indent=2)
            
            print(f"Created mapping: {top_targets}")
            print(f"Saved to dom_stage1_mapping.json")
            break
