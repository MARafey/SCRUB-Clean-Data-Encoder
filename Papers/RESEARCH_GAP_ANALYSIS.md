# Research Gap Analysis — Clean Data Encoder (SCRUB)

Compiled from a full pass over the 28 downloaded reference papers in `Papers/`
(Fellegi & Sunter 1969 excluded — paywalled, not downloaded). For each paper:
technique, primary limitation, the exact section/page/table that evidences it,
and the specific purpose it fails for. Synthesized into research gaps, a
research problem statement, and how SCRUB's approach addresses each gap.

---

## 1. Per-paper limitations with citations

### A. Instruction-conditioned / general-purpose embedding methods

**01 — INSTRUCTOR** (Su et al., arXiv:2212.09741, 2022)
Prepends a task/domain instruction to text before encoding (T5/GTR backbone,
trained on MEDI, 330 datasets).
*Limitation:* Instruction taxonomy is fixed to **Text Type / Task Objective /
Domain** only — Section 2.3 & Table 1 (p.4): "Text Type specifies the type of
input text... Task Objective (Optional) describes the objective... Domain
(Optional) describes the task domain." No slot exists for a *noise/corruption*
description, and none of the 70 MTEB eval tasks involve dirty-vs-clean
structured record matching.
*Fails for:* conditioning an embedding on **what kind of noise** a record
contains.

**02 — E5-Mistral** (Wang et al., arXiv:2401.00368, 2024)
LLM (Mistral-7B) fine-tuned contrastively on GPT-4-generated synthetic
(query, positive, hard-negative) triples with a `Instruct: {task}\nQuery:`
template (Eq. 1 / Table 14, p.4–5).
*Limitation:* `{task_definition}` slots are brainstormed from retrieval/STS
task types only — never noise-type conditions; zero evaluation on structured
noisy-address-style matching.
*Fails for:* same as above — no noise-type instruction vocabulary in training
data.

**27 — Sentence-BERT** (Reimers & Gurevych, EMNLP 2019)
Siamese/triplet BERT producing a fixed, uninstructed sentence embedding
(Section 3, "Model").
*Fails for:* any instruction- or noise-conditioning at all — there is no
prompt input to the architecture; evaluated only on STS/NLI free text
(Section 8, Conclusion), never structured/tabular records.

**11 — Sudowoodo** (Wang, Li & Wang, ICDE 2022)
Contrastive (SimCLR-style) self-supervised pretraining over serialized
records using generic augmentation operators (synonym swap, span deletion).
*Limitation:* Section VII-F / Table XIV (p.15–16), own error analysis:
"the second mistake of not correcting '3344x33541' into '3344933541'... is
due to multiple valid replacements exist[ing]" and recommends "coupl[ing]
Sudowoodo with a pretrained sequence generation model... to generate and
select high-quality corrections" as future work.
*Fails for:* disambiguating among multiple plausible corrections when the
noise type is not explicitly stated — its "instructions" are task-agnostic
augmentation ops, not natural-language noise descriptions.

**04 — Pretrained Embeddings for Entity Resolution** (Zeakis et al., VLDB 2023)
Benchmarks 12 off-the-shelf embedding models (static + BERT/SBERT) for
blocking/matching, no instruction mechanism.
*Limitation:* Section 2, "Gaps" paragraph (p.2226): "We observe that none of
these works considers the main SentenceBERT models... no work has examined
the performance of BERT models... on blocking."
*Fails for:* any instruction- or noise-type-aware conditioning — purely a
raw-effectiveness/efficiency benchmark of clean serialized text.

### B. Entity matching / record linkage

**03 — WDC Products** (Peeters, Der & Bizer, EDBT 2024)
Real-world product-matching benchmark, 27 difficulty variants.
*Limitation:* Section 5.2 "Dimension Seen/Unseen" (p.8): contrastively
pre-trained matchers (R-SupCon) drop ~25% F1 on entities unseen at train
time — "the contrastive pre-training thus severely limits the generalization
ability of this approach."
*Fails for:* generalizing to a corrupted record never seen in training
without retraining.

**05 — Heterogeneity in Entity Matching Survey** (arXiv:2508.08076, 2025)
Taxonomy of representation vs. semantic heterogeneity + targeted stress tests.
*Limitation:* Section 1 (p.2): "High-performing models trained on clean,
benchmark datasets often fail to generalize when exposed to messy, noisy,
and heterogeneous data found in practice... existing benchmarks often mask
these limitations."
*Fails for:* no surveyed method conditions matching on noise *type*
(formatting vs. lexical vs. missing) — descriptive taxonomy only, not a
trainable mechanism.

**09 — fuzzylink** (Ornstein, Political Analysis 2025/2026)
Active-learning logistic regression over frozen pretrained-embedding cosine
similarity + Jaro-Winkler + LLM zero-shot labeling.
*Limitation:* p.4/7: "I focus on applications with a single fuzzy matching
variable... and conclude by discussing how one might extend the procedure to
multiple fuzzy matching variables." Also footnote 1, p.5, on embedding
knowledge-cutoff staleness.
*Fails for:* jointly-noisy multi-field records (an address has several
simultaneously-corrupted attributes) and instruction-conditioning of the
embedding itself — similarity is purely generic/frozen.

**10 — Ditto** (Li et al., PVLDB 2020)
Fine-tunes a PLM as a binary matcher on serialized record pairs.
*Limitation:* Section 2.2 (p.3): "fine-tune a pretrained LM for the EM task
with a labeled training dataset consisting of positive and negative pairs."
*Fails for:* producing a reusable, general-purpose embedding space — Ditto's
output is a per-instance binary decision, not a vector usable for large-scale
nearest-neighbor matching.

**21 — DeepER** (Ebraheem et al., PVLDB 2018)
Composes tuple vectors from pretrained (static) word embeddings.
*Limitation:* Section 3.3 (p.6): "the worst case scenario is specialized
database where no auxiliary resources are available... any machine learning
approach is doomed to fail unless one provides hand crafted features or a
substantially large number of training examples."
*Fails for:* out-of-vocabulary / noisy surface forms not covered by generic
pretrained embeddings.

**22 — Magellan** (Konda et al., PVLDB 2016)
End-to-end EM toolkit over hand-engineered similarity features.
*Limitation:* Section 2.2, item 4 (p.1199): "in many EM scenarios users
simply do not know what to do... No EM system that we have examined provides
such guidance."
*Fails for:* automatic, instruction-conditioned handling of noisy records —
requires a human to iteratively re-engineer features per dataset.

**26 — DeepMatcher** (Mudgal et al., SIGMOD 2018)
Systematic DL-architecture comparison for EM.
*Limitation:* Section 6, challenge C3 (p.7): "DL models often require large
amounts of training data to achieve good performance." Its "dirty" EM variant
only shuffles attribute values between cells — one narrow synthetic noise
type, not typo/formatting noise.
*Fails for:* typo/formatting-level noise robustness and low-label training.

**18 — ZeroER** (Wu et al., SIGMOD 2020)
Unsupervised Gaussian-mixture model over Magellan-style similarity features.
*Limitation:* Section 7 (p.14): "Currently, ZeroER treats feature engineering
as a black box and uses features generated by Magellan. Designing automatic
feature engineering solution for ER in general is an interesting future
work."
*Fails for:* end-to-end learned/instruction-conditioned representations —
zero-label gain still depends on hand-crafted features as input.

### C. Classical / rule-based & statistical cleaning systems

**13 — Raha** (Mahdavi et al., SIGMOD 2019)
Ensemble of hand-designed detectors (outlier/pattern/rule/KB) + semi-supervised
classifier.
*Limitation:* Abstract (p.1) and §1: "The focus of this work is on error
detection" — no correction step at all.
*Fails for:* correcting/mapping a corrupted record to its clean canonical
twin.

**14 — DataWig** (Biessmann et al., JMLR 2019)
Per-column deep-learning imputation.
*Limitation:* §4 Conclusion (p.4): "the present version of our software does
not impute free form text or images, an interesting topic for future
research is using generative models for these types of data." Best correction
F1 on AddressTable is only ~0.29 (per `08`, Table 4).
*Fails for:* generating/matching free-form corrupted text (a full noisy
address string), not just discrete categorical values.

**15 — HoloClean** (Rekatsinas et al., VLDB 2017)
Probabilistic factor-graph repair over hand-authored constraints/dictionaries.
*Limitation:* §2.2 (p.4): "HoloClean treats error detection as a black
box... Error detection is out of scope for this paper." Correction F1 on
AddressTable is near-zero (~0.001–0.004, per `08` Table 4) despite decent
detection.
*Fails for:* signal-free semantic matching — repair quality is bounded by
externally supplied constraints/dictionaries, with no learned embedding space.

**16 — HoloDetect** (Heidari et al., SIGMOD 2019)
Few-shot detection via noisy-channel data augmentation.
*Limitation:* §5.1 (p.6) restricts noise transformations to "Add characters /
Remove characters / Exchange characters" — shallow character-level only.
*Fails for:* producing a corrected canonical output or embedding-space
alignment; detection-only, narrow noise model.

**17 — ActiveClean** (Krishnan et al., PVLDB 2016)
SGD-prioritized human-in-the-loop cleaning for downstream models.
*Limitation:* §3.3 (p.3): "The Cleaner is user-specified and executes C(·)
for each sample record" — the actual cleaning step is manual; its optional
feature signal is TF-IDF only (word-level).
*Fails for:* automated, instruction-conditioned semantic correction — cleaning
itself is not learned.

**23 — NADEEF** (Dallachiesa et al., SIGMOD 2013)
Unified rule-programming interface for heterogeneous constraint types.
*Limitation:* §3 (p.4): repairs originate entirely from user-authored `Rule`
classes implementing `vio()`/`fix()`.
*Fails for:* unsupervised/embedding-based matching without hand-authored
rules — every error type must be explicitly coded in advance.

### D. LLM-based data cleaning & entity matching

**06 — Data Cleaning Using LLMs / Cocoon** (Zhang, Huang, Wu, arXiv:2410.15547, 2024)
Per-error-type statistical detection + LLM-generated SQL mapping.
*Limitation:* §2.1.1 (p.2): LLM evaluates "one batch at a time" (≤1000
values); output is one-off SQL, never an embedding.
*Fails for:* a reusable/index-able vector representation for large-scale
matching.

**07 — TableEG** (arXiv:2507.10934, 2025)
Fine-tuned LLM that *generates* synthetic errors for benchmarking.
*Limitation:* §3.2/Fig.2 (p.4): pipeline outputs structured JSON edit triples,
not embeddings; scoped to error generation, not matching.
*Fails for:* instruction-conditioned embedding retrieval (out of scope by
design).

**12 — Can Foundation Models Wrangle Your Data?** (Narayan et al., PVLDB 2022)
Zero/few-shot GPT-3 prompting for EM, imputation, error detection.
*Limitation:* §4.3 (p.742): "FMs can be brittle to subtle variations in
prompt templates... a minor modification results in an average 9.4 F1 point
performance gap." Matching is pairwise generative — no shared representation.
*Fails for:* scalable large-corpus matching (quadratic pairwise cost) and
prompt-invariance.

**19 — Using ChatGPT for Entity Matching** (Peeters & Bizer, ADBIS 2023)
Zero/few-shot ChatGPT pairwise product matching.
*Limitation:* Table 2 (p.5): swapping "the same" → "equivalent" changes
results across all datasets; Table 4/Cost Analysis (p.7–8): 20 in-context
examples raise cost ~1400%.
*Fails for:* cost-scalable, format-robust matching with a reusable
representation.

**20 — Entity Matching using LLMs** (Peeters, Steiner, Bizer, EDBT 2025)
Benchmarks hosted/open LLMs for EM across prompting strategies.
*Limitation:* §3 "Sensitivity" (p.4): non-GPT-4 models show prompt-sensitivity
std. dev. 6.18–18.54 (Table 3); §2: "No single best prompt... depends on the
model/dataset combination."
*Fails for:* a prompt-invariant, index-able matching representation.

**24 — Table-GPT** (Li et al., arXiv:2310.09263, 2024)
Continues training GPT-3.5 on synthesized table-task triples.
*Limitation:* §1/Fig.1 (p.1–2): base models score as low as 0.26 accuracy on
column-finding tests and "fail on a surprisingly large fraction (up to 74%)"
of simple column-value lookups.
*Fails for:* reliable structured/columnar understanding needed for consistent
embedding-based matching; output remains per-query generation, not a vector
index.

**25 — Jellyfish** (Zhang et al., arXiv:2312.01678, EMNLP 2024)
Instruction-tunes local 7–13B LLMs for DP tasks.
*Limitation:* §2.2 (p.4): "LLMs often require substantial computational
resources... the input to the LLM is often instance-by-instance... results
may exhibit inconsistency across different instances."
*Fails for:* scalable, cross-record-consistent matching — no persistent
representation to amortize cost or enforce consistency.

**28 — BoostER** (Li et al., WWW Companion 2024)
Entropy-guided selection of LLM verification queries over a base ER tool.
*Limitation:* §1 (p.1): "posing all matching questions could result in
substantial costs"; §2.2: explicit per-question token-price function.
*Fails for:* eliminating (not just budgeting) per-pair LLM query cost — no
embedding-based alternative that avoids pairwise querying.

**29 — GPT-4 Scrambled Text Correction** (Cao et al., arXiv:2311.18805, 2023)
Benchmarks GPT-4 recovering character-scrambled words via generation.
*Limitation:* "Limitations" section (final page): "we only investigate the
influence of scrambling the letter-order in words... it is difficult to
conclude the reason why some LLMs are capable of handling these tasks."
*Fails for:* noise-type generality and any embedding-based downstream
retrieval — single narrow noise type, generative-only, mechanism unexplained.

### E. Target dataset paper (the benchmark this project targets)

**08 — Clean Me If You Can** (Ahmadi et al., arXiv:2606.31983, 2026)
Introduces AddressTable (up to 9.3M records, 12-category corruption taxonomy,
§5.2.1 pp.6–7) and benchmarks 6 techniques — **Horizon** (rules), **UniClean**
(constraints), **HoloClean** (factor graph), **Raha/Baran** (semi-supervised),
**DataWig** (imputation), **Glock's** (LLM prompting) — in §6, Table 4/5
(pp.10–11).

Correction-F1 results (Table 4, POI slices): Horizon 0.12–0.40, UniClean
~0.01, HoloClean 0.001–0.004, DataWig 0.27–0.29, Baran (best) ~0.46–0.69
avg, Glock's ~0.42 avg — but Glock's costs "approximately $650" and 8 hours
per 1M records (§6.4.1–6.4.2, p.11), a stated scalability failure.

**Explicit takeaway** (Contribution 2, Introduction, p.1): *"None of the
evaluated data cleaning approaches can reliably clean the dataset or even its
slices."*
**Future-work statement** (§6.5 Key Takeaways / §7 Conclusions, p.12):
*"Future cleaning research would benefit from approaches that take the
relevant metadata into consideration and recalibrate their detection and
correction strategies."*

Confirmed: **no embedding-based or instruction-conditioned method appears
anywhere in this paper** — not in Related Work (§2), not in Baselines (§6.2),
not in the future-work discussion. This is the direct evidentiary gap this
project targets.

---

## 2. Synthesized research gaps

Reading across all 28 papers, four **recurring, independently-evidenced**
gaps emerge — each paper above maps to at least one of these:

**Gap 1 — No noise-*type* conditioning exists in any instruction-capable
embedding model.**
INSTRUCTOR (01) and E5-Mistral (02) prove instructions can steer an
embedding space, but their instruction vocabularies are fixed to task/domain
descriptors — never "this field has a typo" / "this field is missing" / "this
field has inconsistent capitalization." Sudowoodo (11) independently shows
*why* this matters: without an explicit noise description, its model cannot
disambiguate between multiple plausible corrections (its own error analysis,
§VII-F). No paper combines an explicit, natural-language noise-type
instruction with an embedding-producing encoder.
> **Example failure (Sudowoodo, §VII-F, Table XIV):** the model is asked to
> correct the corrupted string `3344x33541` and fails — it does not fix the
> `x` to `9` — because, in the paper's own words, "multiple valid
> replacements exist" and nothing tells the model *which kind* of corruption
> to expect (a single-character substitution vs. an OCR error vs. a
> transposed digit). An instruction such as *"this field may contain a
> single mistyped digit"* would disambiguate the correction outright, but no
> such noise-type slot exists in Sudowoodo's augmentation-only conditioning,
> nor in INSTRUCTOR's/E5-Mistral's task/domain-only instruction schema.

**Gap 2 — Correction-capable techniques either don't produce embeddings, or
embedding-capable techniques don't correct.**
Classical/statistical cleaners (Raha 13, HoloDetect 16 — detection only;
NADEEF 23, HoloClean 15 — rule/constraint-bounded correction) and LLM-based
cleaners (06, 07, 12, 19, 20, 24, 25, 28, 29 — generative text-in/text-out,
no reusable vector) sit on one side. Embedding-producing methods (Ditto 10 —
binary classifier not a space; DeepER 21, VLDB benchmark 04, SBERT 27 — no
noise-awareness; Sudowoodo 11 — augmentation-only) sit on the other. **Gap:**
no technique produces a reusable, index-able embedding space *and* performs
noise-aware correction/matching in the same step.
> **Example failure (HoloClean, on AddressTable, `08` Table 4):** on the
> `library` POI slice, HoloClean reaches a respectable **0.80 detection F1**
> (it can flag `"123 Mian St, Sprngfield"` as containing errors) but only
> **0.001 correction F1** — it essentially never outputs the right fix. Per
> its own §2.2: "HoloClean treats error detection as a black box... Error
> detection is out of scope for this paper" — repair depends entirely on
> externally supplied denial constraints/dictionaries that don't exist for
> free-text fields like `road` or `name`, so a semantically obvious fix
> ("Mian"→"Main", "Sprngfield"→"Springfield") is never attempted. Conversely
> Ditto (10) *can* be trained to say "these two addresses match," but its
> output is a single binary decision per fine-tuned pair — it never produces
> a vector you could look up "the nearest 5 canonical addresses to this
> corrupted one" with.

**Gap 3 — Scalability collapses whenever accuracy is acceptable.**
Every technique that gets reasonably close to usable correction quality on
real noisy data does so by paying a per-record or per-pair cost that doesn't
scale: Glock's/LLM-prompting costs ~$650 and 8h/1M records on AddressTable
itself (08, §6.4); Can Foundation Models Wrangle Your Data? (12) and
ChatGPT-EM (19, 20) require one LLM call per candidate pair with results
sensitive to prompt wording; BoostER (28) exists specifically to *budget*
(not eliminate) this cost. Embedding-based matching is the only approach
family that is inherently sub-quadratic (encode once, compare vectors) — but
per Gap 1/2, no existing embedding method is both noise-aware and accurate
enough to exploit that advantage.
> **Example failure (Glock's LLM-prompting baseline, on AddressTable, `08`
> §6.4.1–6.4.2):** Glock's reaches the second-best correction F1 in the whole
> benchmark (~0.42 avg, behind only Baran), but doing so requires **one
> Gemini API call per record**. The paper reports this "already requir[es]
> 8 hours to process 1M records" and "a single run on the full [9.3M-record]
> dataset would cost approximately $650." The same qualitative pattern
> repeats in ChatGPT-EM (19): adding 20 in-context examples per pair to
> improve accuracy raises per-decision cost by "nearly 1400%" (Table 4).
> Accuracy and scalability trade off directly whenever the underlying
> mechanism is per-instance LLM inference rather than a precomputed vector
> index.

**Gap 4 — Generalization to unseen/noisy surface forms is unsolved without
either retraining or external knowledge.**
WDC Products (03) shows contrastively pretrained matchers lose ~25% F1 on
unseen entities; DeepER (21) explicitly states "any machine learning approach
is doomed to fail" without external resources for out-of-vocabulary/noisy
surface forms; the Heterogeneity survey (05) confirms this is systemic across
the field and says current benchmarks mask it. fuzzylink (09) sidesteps this
with a frozen pretrained embedding plus an LLM classifier, but is explicitly
scoped to single-field fuzzy matching, not multi-field jointly-corrupted
records like addresses.
> **Example failure (WDC Products, §5.2 "Dimension Seen/Unseen", p.8):** a
> contrastively pretrained matcher (R-SupCon) that performs well during
> evaluation on product entities it saw during training loses **~25 F1
> points** the moment it is tested against entities it has never seen before
> — the paper attributes this directly to the pre-training objective:
> "the contrastive pre-training thus severely limits the generalization
> ability of this approach." Applied to addresses, this predicts the same
> failure mode: a matcher trained on US addresses would be expected to
> degrade sharply on a previously-unseen German or Russian address format,
> exactly the cross-locale slices (`de-slice`, `ru-slice`) the target
> AddressTable paper (08) tests against.

**Anchor fact:** The paper that owns this project's actual target dataset
(08, Clean Me If You Can) states outright that *none* of the six techniques
it benchmarks — spanning rules, statistics, semi-supervised classifiers,
imputation, and LLM prompting — "can reliably clean the dataset," and never
considers an embedding-based approach as an alternative.

---

## 3. Research problem statement

> **Given a corrupted structured record and an explicit natural-language
> description of the kind of noise it may contain, can a *single*,
> instruction-conditioned embedding encoder — asymmetrically applied only to
> the noisy query side — pull that record's embedding close enough to its
> clean canonical twin's (noise-free) embedding to support accurate,
> scalable nearest-neighbor matching/correction, in a way that no existing
> rule-based, statistical, feature-engineered, LLM-prompting, or
> plain/task-instructed embedding technique achieves?**

This directly targets the intersection of Gaps 1–3: it requires (a) a noise-
*type* instruction vocabulary (missing from 01/02/11), (b) a single encoder
that is simultaneously correction-capable and embedding-producing (missing
from the 10/16/23/15 vs. 06/07/12/19/20/24/25/28/29 split), and (c) a
sub-quadratic, index-able matching cost (the one advantage embeddings have
over LLM-prompting, but never combined with noise-conditioning in the
literature). It is evaluated on exactly the dataset (08) whose own authors
confirm no existing technique clears this bar.

## 4. How the SCRUB approach solves it

Per [code/README.md](../code/README.md) and `code/instructions.py`/`model.py`:

- **Addresses Gap 1** — `INSTRUCTION_BANK` (`code/instructions.py`) defines
  *noise-type-descriptive* instructions (e.g. `noise_domain`, `noise_terse`,
  `noise_specific`), not task-type ones — the exact instruction category
  INSTRUCTOR/E5-Mistral never populate. `query_instruction` is applied only
  to the corrupted side; `doc_instruction` stays `none` so the clean
  canonical index remains instruction-free and reusable, giving an
  *asymmetric* conditioning mechanism absent from every paper in Section A.

- **Addresses Gap 2** — SCRUB is a single encoder (`model.py`: trainable
  word-embedding + char-CNN hybrid token representation feeding N pre-norm
  Transformer blocks, masked mean pooling) trained end-to-end with a
  bidirectional in-batch InfoNCE contrastive loss (`losses.py`) directly on
  (noisy, clean) pairs. There is no separate detection stage and no
  generative text-correction stage — the corrected identity *is* the nearest
  canonical embedding, unifying "produces embeddings" and "is noise-aware"
  in one model, which no paper in Sections A–D does simultaneously.

- **Addresses Gap 3** — Because matching is nearest-neighbor lookup in a
  fixed-dimension vector space against a pre-computed canonical index (no
  LLM call per record, no per-pair prompt), inference cost is encode-once /
  compare-many — sub-quadratic, unlike Glock's, BoostER, or any ChatGPT-EM
  variant (06, 07, 12, 19, 20, 24, 25, 28).

- **Partially addresses Gap 4** — because SCRUB is trained from scratch only
  on this project's own (noisy, clean) pairs plus the instruction bank text
  (`tokenizer.py`), it has no dependency on external pretrained-embedding
  vocabulary coverage (unlike DeepER 21 or fuzzylink 09); generalization to
  truly unseen surface forms is still bounded by training-pair diversity —
  this is an open question for the project, not yet a proven advantage.

**Known caveat to carry forward (do not overstate the current result):** per
`code/README.md`, a TF-IDF/char-n-gram baseline (`baseline_lexical.py`) still
**outperforms** SCRUB on every tested AddressTable slice as of the last
recorded run. The research gap and problem statement above are validated by
the literature; SCRUB's mechanism is a structurally sound answer to that gap,
but **empirical superiority over even a non-embedding lexical baseline is
not yet demonstrated** and remains the open next milestone before claiming
the gap is closed.
