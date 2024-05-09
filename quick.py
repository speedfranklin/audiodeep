import soundata
import pandas as pd
from pathlib import Path

# learn wich datasets are available in soundata
#print(soundata.list_datasets())

# choose a dataset and download it
dataset = soundata.initialize('urbansound8k', data_home='data')
dataset.download()

# get annotations and audio for a random clip
example_clip = dataset.choice_clip()
print(f"{example_clip.clip_id}")
tags = example_clip.tags
y, sr = example_clip.audio

# ----------------------------
# Prepare training data from Metadata file
# ----------------------------
download_path = Path.cwd()/'data'

# Read metadata file
metadata_file_path = download_path/'metadata'/'UrbanSound8K.csv'
print(f"metadata_file_path: {metadata_file_path}")
df = pd.read_csv(metadata_file_path)
df.head()

# Construct file path by concatenating fold and file name
df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)

# Take relevant columns
df = df[['relative_path', 'classID']]
df.head()