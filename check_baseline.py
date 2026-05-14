#!/usr/bin/env python3
import json
from pathlib import Path

results_file = Path('dom_brand_stage1_results.jsonl')
if results_file.exists():
    with open(results_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data.get('cmd') == 'mc-train' and data.get('algorithm') == 'random_forest':
                class_names = data.get('class_names', [])
                class_hist_raw = data.get('class_hist_raw', {})
                print('Baseline class distribution:')
                sorted_hist = sorted(class_hist_raw.items(), key=lambda x: x[1], reverse=True)
                total = sum(class_hist_raw.values())
                for cls, cnt in sorted_hist:
                    pct = (cnt / total * 100) if total > 0 else 0
                    print(f'{cls:25} : {cnt:9,} ({pct:5.1f}%)')
                print(f'\nTotal: {total:,}')
                break
