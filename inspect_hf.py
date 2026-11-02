from datasets import load_dataset
import numpy as np

print("Loading HF dataset...")
ds = load_dataset('asolodin/rescuenet-damaged-road-clips', split='train', streaming=True)
item = next(iter(ds))

mask = item['label']
mask_np = np.array(mask)

print("Unique label values in HF item:", np.unique(mask_np))
