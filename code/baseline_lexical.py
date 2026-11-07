"""Dumb lexical baselines on the exact same split as the embedding runs.

If TF-IDF character n-grams - no neural model, no instruction, no GPU - match
the trained encoder's numbers, then the headline accuracy is a property of the
task, not of the method, and the ablation has no room to show anything.
"""
import numpy as np, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from config import ExperimentConfig
from data import load_dataset
import metrics as M

import sys
DS = sys.argv[1] if len(sys.argv)>1 else "full_051"
cfg = ExperimentConfig(dataset=DS, run_name="lexical_baseline")
data = load_dataset(cfg, verbose=False)
split = data.splits["test"]
qrows = split.indices
print(f"dataset={DS}  canonical={len(data.canonical_texts)}  test queries={len(qrows)}")

for name, kw in [
    ("char 3-5gram TF-IDF", dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
    ("word unigram TF-IDF", dict(analyzer="word", ngram_range=(1, 1), min_df=1)),
]:
    t0 = time.time()
    vec = TfidfVectorizer(**kw).fit(data.canonical_texts + data.dirty_texts)
    C = normalize(vec.transform(data.canonical_texts)).astype(np.float32)
    Q = normalize(vec.transform([split.dirty_texts[i] for i in range(len(qrows))])).astype(np.float32)

    ranks = np.zeros(len(qrows), dtype=np.int64)
    top1s = np.zeros(len(qrows), dtype=np.float32)
    top1i = np.zeros(len(qrows), dtype=np.int64)
    for s in range(0, Q.shape[0], 256):
        blk = Q[s:s+256]
        sims = (blk @ C.T).toarray()
        gold = qrows[s:s+256]
        gs = sims[np.arange(len(gold)), gold][:, None]
        ranks[s:s+256] = (sims > gs).sum(axis=1) + 1
        top1i[s:s+256] = sims.argmax(axis=1)
        top1s[s:s+256] = sims.max(axis=1)

    r = M.retrieval_metrics(ranks, [1, 5, 10])
    sw = M.sweep_thresholds(data.clean_df, data.dirty_df, qrows, top1i, top1s,
                            data.fields, [round(0.05*i, 2) for i in range(20)])
    print(f"\n{name}  ({time.time()-t0:.1f}s, {C.shape[1]} features)")
    print(f"   Recall@1={r['recall@1']:.4f}  Recall@5={r['recall@5']:.4f}  MRR={r['mrr']:.4f}")
    print(f"   correction F1={sw['best_correction']['correction_f1']:.4f}"
          f"  detection F1={sw['best_detection']['detection_f1']:.4f}")
