"""
Extract metadata for batch 1 and save it to a new Parquet file.
"""
from pathlib import Path
import pyarrow.parquet as pq

path = Path('data/train_meta.parquet')
table = pq.read_table(path)
df = table.to_pandas()

filtered_df = df[df['batch_id'] == 1]
new_path = Path('data/train_meta_batch_1.parquet')
filtered_df.to_parquet(new_path, index=False)
