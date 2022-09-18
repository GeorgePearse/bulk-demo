import pandas as pd
from umap import UMAP
# pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer

# Load the universal sentence encoder
# Stay well clear of the direct Hugging Face API which is grim
sample_size = 30000
model = SentenceTransformer('all-mpnet-base-v2')

embeddings = model.encode(sentences[:sample_size])

# Load original dataset
df = pd.read_csv("original.csv")
sentences = df["text"]

# Calculate embeddings 
X =  model.encode(sentences)

# Reduce the dimensions with UMAP
umap = UMAP(n_components=2, verbose=True)
X_tfm = umap.fit_transform(embeddings)

# Apply coordinates
df = df.iloc[:sample_size]
df['x'] = X_tfm[:, 0]
df['y'] = X_tfm[:, 1]
df.to_csv("ready.csv")
