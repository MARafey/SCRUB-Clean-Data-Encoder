"""
Contrastive objective for noisy -> clean alignment.

Same shape as INSTRUCTOR's training loss: in-batch softmax over cosine
similarities, optionally computed in both directions (dirty->clean and
clean->dirty) and averaged. The positives are the row-aligned pairs; every
other clean record in the batch is a negative.

AddressTable has real duplicate canonical records (the same real-world
address, e.g. a chain entity, repeated verbatim as separate rows - roughly
17% of russia_065's canonical set). When two batch positions i != j share an
identical clean text, treating j as a negative for i's positive pair is
wrong: it is not a hard negative, it is the same entity, and penalising the
model for scoring them equally corrupts the gradient. `duplicate_mask` lets
the caller exclude exactly those (i, j) pairs from the softmax denominator in
both directions, without touching the true diagonal positive.
"""
from __future__ import annotations

from typing import Optional

import tensorflow as tf


def info_nce(query_emb: tf.Tensor,
             doc_emb: tf.Tensor,
             temperature: float = 0.05,
             bidirectional: bool = True,
             duplicate_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
    """Cross-entropy over in-batch cosine similarities.

    Args:
        query_emb: [B, H] L2-normalised embeddings of the dirty records.
        doc_emb:   [B, H] L2-normalised embeddings of their clean twins.
        temperature: softmax temperature; lower sharpens the distribution.
        bidirectional: also score clean->dirty and average the two losses.
        duplicate_mask: optional [B, B] bool tensor, symmetric, False on the
            diagonal. True at (i, j) means batch positions i and j are the
            same real-world clean entity (duplicate canonical text), so (i,
            j) and (j, i) are excluded from the negative set in both
            directions rather than scored as a mismatch.

    Returns:
        Scalar loss.
    """
    query_emb = tf.cast(query_emb, tf.float32)
    doc_emb = tf.cast(doc_emb, tf.float32)

    # Embeddings are already normalised by the encoder, but a second pass keeps
    # this function correct if it is ever called with raw vectors.
    query_emb = tf.math.l2_normalize(query_emb, axis=-1)
    doc_emb = tf.math.l2_normalize(doc_emb, axis=-1)

    logits = tf.matmul(query_emb, doc_emb, transpose_b=True) / temperature
    if duplicate_mask is not None:
        neg_fill = tf.fill(tf.shape(logits), logits.dtype.min)
        logits = tf.where(duplicate_mask, neg_fill, logits)
    labels = tf.range(tf.shape(logits)[0])

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    )

    if bidirectional:
        # duplicate_mask is symmetric by construction (shared clean text is
        # a symmetric relation), so the same mask applies to the transpose.
        loss_rev = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels, tf.transpose(logits)
            )
        )
        loss = 0.5 * (loss + loss_rev)

    return loss


def batch_alignment_stats(query_emb: tf.Tensor,
                          doc_emb: tf.Tensor,
                          duplicate_mask: Optional[tf.Tensor] = None) -> dict:
    """Diagnostics logged alongside the loss.

    pos_sim    mean cosine similarity of the true pairs
    neg_sim    mean cosine similarity of the hardest in-batch negative
    margin     pos_sim - neg_sim; the quantity training is really moving
    accuracy   fraction of rows whose true clean twin is the in-batch argmax
               (or a duplicate of it, when duplicate_mask is given)

    duplicate_mask excludes same-entity batch pairs from both the hardest-
    negative search and the argmax-accuracy check, for the same reason it is
    excluded from the loss in `info_nce`: a duplicate is not a negative.
    """
    query_emb = tf.math.l2_normalize(tf.cast(query_emb, tf.float32), axis=-1)
    doc_emb = tf.math.l2_normalize(tf.cast(doc_emb, tf.float32), axis=-1)

    sims = tf.matmul(query_emb, doc_emb, transpose_b=True)
    batch = tf.shape(sims)[0]
    eye = tf.eye(batch, dtype=tf.bool)
    excluded = eye if duplicate_mask is None else tf.logical_or(eye, duplicate_mask)

    pos = tf.boolean_mask(sims, eye)
    neg_only = tf.where(excluded, tf.fill(tf.shape(sims), sims.dtype.min), sims)
    hardest_neg = tf.reduce_max(neg_only, axis=1)

    argmax_idx = tf.cast(tf.argmax(
        tf.where(duplicate_mask, tf.fill(tf.shape(sims), sims.dtype.min), sims)
        if duplicate_mask is not None else sims,
        axis=1), tf.int32)
    correct = tf.equal(argmax_idx, tf.range(batch))

    return {
        "pos_sim": tf.reduce_mean(pos),
        "neg_sim": tf.reduce_mean(hardest_neg),
        "margin": tf.reduce_mean(pos - hardest_neg),
        "in_batch_accuracy": tf.reduce_mean(tf.cast(correct, tf.float32)),
    }
