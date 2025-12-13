#!/usr/bin/env python3
"""
Tutorial script: Using Scikit-bio with real biological data
Steps 2–6: Load dataset, wrap into scikit-bio objects, analyze, compute diversity, visualize
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import skbio
from skbio.diversity import alpha_diversity

# ----------------------------
# Step 2. Load dataset
# ----------------------------
# Example: 16S rRNA sequences from Greengenes (subset)
url = "https://raw.githubusercontent.com/biocore/scikit-bio/master/skbio/data/sequence/gg_13_8_otus_99.fasta.gz"

# Read FASTA file directly with scikit-bio
seqs = list(skbio.io.read(url, format="fasta", constructor=skbio.DNA))

print(f"Loaded {len(seqs)} sequences")
print("First sequence:", seqs[0])

# ----------------------------
# Step 3. Wrap into scikit-bio objects
# ----------------------------
# Already DNA objects; let’s compute GC content for each
gc_contents = [s.gc_content() for s in seqs]

df = pd.DataFrame({
    "id": [s.metadata["id"] for s in seqs],
    "gc_content": gc_contents
})

print(df.head())

# ----------------------------
# Step 4. Sequence analysis
# ----------------------------
example_seq = seqs[0]
print("Example sequence ID:", example_seq.metadata["id"])
print("GC content:", example_seq.gc_content())
print("Reverse complement:", str(example_seq.reverse_complement()))

# ----------------------------
# Step 5. Diversity metrics
# ----------------------------
# For demonstration, treat k-mer counts as "species abundances"
def kmer_counts(seq, k=4):
    counts = {}
    for i in range(len(seq) - k + 1):
        kmer = str(seq[i:i+k])
        counts[kmer] = counts.get(kmer, 0) + 1
    return counts

# Build abundance table for first 20 sequences
abundance = []
ids = []
for s in seqs[:20]:
    counts = kmer_counts(s, k=4)
    abundance.append([counts.get(k, 0) for k in sorted(counts.keys())])
    ids.append(s.metadata["id"])

# Compute Shannon diversity
shannon = alpha_diversity("shannon", abundance, ids=ids)
print(shannon)

# ----------------------------
# Step 6. Visualization
# ----------------------------
sns.histplot(df["gc_content"], kde=True)
plt.title("GC Content Distribution of 16S rRNA Sequences")
plt.xlabel("GC Content")
plt.ylabel("Frequency")
plt.show()

