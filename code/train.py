"""
Contrastive fine-tuning with resumable checkpoints and full TensorBoard logging.

Checkpointing contract
----------------------
Two managers write into the run directory:

    checkpoints/       rolling, last N steps, for resuming an interrupted run
    checkpoints_best/  single checkpoint, overwritten only when the validation
                       metric named by cfg.best_metric improves

Optimiser slots, global step, epoch and the best-metric-so-far all live inside
the checkpoint, so relaunching the same run name picks up exactly where it
stopped: same step counter, same optimiser momentum, same best score to beat.
Pass resume=False to start over instead.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from config import ExperimentConfig, parse_overrides
from data import AddressData, load_dataset, make_train_arrays
from instructions import describe, get_instruction, is_per_record
from losses import batch_alignment_stats, info_nce
from model import InstructionEncoder, build_encoder


# --------------------------------------------------------------------------
# Learning rate schedule
# --------------------------------------------------------------------------
class WarmupLinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup then linear decay - stabilises early training for a
    from-scratch Transformer, whose randomly-initialised attention layers are
    otherwise prone to large, destabilising gradients in the first steps."""

    def __init__(self, peak_lr: float, total_steps: int, warmup_steps: int):
        super().__init__()
        self.peak_lr = float(peak_lr)
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = max(int(warmup_steps), 1)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)
        warming = self.peak_lr * step / warm
        decaying = self.peak_lr * tf.maximum(
            0.0, (total - step) / tf.maximum(total - warm, 1.0)
        )
        return tf.where(step < warm, warming, decaying)

    def get_config(self):
        return {"peak_lr": self.peak_lr, "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps}


# --------------------------------------------------------------------------
# Checkpoint plumbing
# --------------------------------------------------------------------------
class TrainState:
    """Everything that has to survive a restart."""

    def __init__(self, encoder: InstructionEncoder, optimizer):
        self.step = tf.Variable(0, dtype=tf.int64, trainable=False, name="step")
        self.epoch = tf.Variable(0, dtype=tf.int64, trainable=False, name="epoch")
        self.best_metric = tf.Variable(-1.0, dtype=tf.float32, trainable=False,
                                       name="best_metric")
        self.checkpoint = tf.train.Checkpoint(
            model=encoder,
            optimizer=optimizer,
            step=self.step,
            epoch=self.epoch,
            best_metric=self.best_metric,
        )


def _managers(state: TrainState, cfg: ExperimentConfig):
    latest = tf.train.CheckpointManager(
        state.checkpoint, cfg.ckpt_dir,
        max_to_keep=cfg.max_checkpoints_to_keep,
        checkpoint_name="ckpt",
    )
    best = tf.train.CheckpointManager(
        state.checkpoint, cfg.best_ckpt_dir,
        max_to_keep=1,
        checkpoint_name="best",
    )
    return latest, best


def restore_encoder(encoder: InstructionEncoder,
                    cfg: ExperimentConfig,
                    prefer_best: bool = True) -> Optional[str]:
    """Restore model weights only - used by evaluate.py, no optimiser needed."""
    ckpt = tf.train.Checkpoint(model=encoder)
    for directory in ([cfg.best_ckpt_dir, cfg.ckpt_dir] if prefer_best
                      else [cfg.ckpt_dir, cfg.best_ckpt_dir]):
        path = tf.train.latest_checkpoint(directory)
        if path:
            ckpt.restore(path).expect_partial()
            print(f"[ckpt] restored weights from {path}")
            return path
    print("[ckpt] no checkpoint found - using the random-init encoder as-is")
    return None


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def _make_train_step(encoder: InstructionEncoder, optimizer, cfg: ExperimentConfig):
    # The tokenize() output shape differs by encoder: the from-scratch
    # encoders (model.py, model_snn.py) return word_ids + char_ids, while the
    # pretrained-backbone encoder (model_pretrained.py) returns input_ids only
    # (a real subword tokenizer, no separate char-CNN path). Build the
    # tf.function signature from whichever this run's encoder actually
    # produces instead of hardcoding the from-scratch shape.
    if cfg.encoder_type == "pretrained":
        sig = {
            "input_ids": tf.TensorSpec([None, None], tf.int32),
            "attention_mask": tf.TensorSpec([None, None], tf.int32),
            "pool_mask": tf.TensorSpec([None, None], tf.int32),
        }
    else:
        sig = {
            "word_ids": tf.TensorSpec([None, None], tf.int32),
            "char_ids": tf.TensorSpec([None, None, None], tf.int32),
            "attention_mask": tf.TensorSpec([None, None], tf.int32),
            "pool_mask": tf.TensorSpec([None, None], tf.int32),
        }

    # See model_pretrained.py's module docstring: a from_config()-built
    # (pretrained_random_init) backbone's variables are not reliably visible
    # through the outer encoder's own .trainable_variables (a Keras 3 / Keras
    # 2 auto-tracking mismatch), so go straight to encoder.backbone for the
    # pretrained path rather than through the outer wrapper.
    trainable_vars = (encoder.backbone.trainable_variables
                      if cfg.encoder_type == "pretrained"
                      else encoder.trainable_variables)

    @tf.function(input_signature=[sig, sig, tf.TensorSpec([None, None], tf.bool)])
    def step_fn(q_batch, d_batch, duplicate_mask):
        with tf.GradientTape() as tape:
            q = encoder(q_batch, training=True)
            d = encoder(d_batch, training=True)
            loss = info_nce(q, d, cfg.temperature, cfg.bidirectional_loss,
                            duplicate_mask=duplicate_mask)
        grads = tape.gradient(loss, trainable_vars)
        if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
            grads, gnorm = tf.clip_by_global_norm(grads, cfg.grad_clip_norm)
        else:
            gnorm = tf.linalg.global_norm(grads)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        stats = batch_alignment_stats(q, d, duplicate_mask=duplicate_mask)
        stats["loss"] = loss
        stats["grad_norm"] = gnorm
        return stats

    return step_fn


def _duplicate_mask(clean_texts: List[str]) -> np.ndarray:
    """[B, B] bool, True at (i, j) i != j when batch rows i and j share the
    same clean canonical text (AddressTable has real duplicate canonical
    records - see losses.py's module docstring). Symmetric, False diagonal.
    """
    n = len(clean_texts)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if clean_texts[i] == clean_texts[j]:
                mask[i, j] = True
                mask[j, i] = True
    return mask


def train(cfg: ExperimentConfig,
          data: Optional[AddressData] = None,
          encoder: Optional[InstructionEncoder] = None) -> Dict:
    from evaluate import evaluate  # imported here to avoid a circular import

    cfg.make_dirs()
    cfg.save()

    data = data or load_dataset(cfg)
    encoder = encoder or build_encoder(cfg, data)

    dirty_train, clean_train = make_train_arrays(data, "train")
    train_row_ids = data.splits["train"].indices  # absolute row ids, needed
                                                    # for per-record instructions
    n_train = len(dirty_train)
    steps_per_epoch = max(1, n_train // cfg.batch_size)

    # The LR schedule's length is fixed the first time a run trains and then
    # reused on every resume, deliberately decoupled from cfg.epochs. Without
    # this, resuming with a larger `epochs` recomputes total_steps from the
    # new value, which reopens a schedule that had already decayed to ~0 back
    # up to a large LR at the current step - a real regression (observed: loss
    # 0.03 -> 2.5 immediately after a 100->200 epoch resume), not just noise.
    # To genuinely extend a schedule rather than just running more (LR-less)
    # epochs, delete lr_schedule.json or start a new --run.
    schedule_path = os.path.join(cfg.run_dir, "lr_schedule.json")
    if os.path.exists(schedule_path):
        with open(schedule_path, "r", encoding="utf-8") as f:
            total_steps = json.load(f)["total_steps"]
        print(f"[train] LR schedule: reusing total_steps={total_steps} from "
              f"{schedule_path} (cfg.epochs={cfg.epochs} controls the loop "
              f"bound only, not schedule length)")
    else:
        total_steps = steps_per_epoch * cfg.epochs
        with open(schedule_path, "w", encoding="utf-8") as f:
            json.dump({"total_steps": total_steps}, f)

    schedule = WarmupLinearDecay(
        cfg.learning_rate, total_steps, int(cfg.warmup_ratio * total_steps)
    )
    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=schedule, weight_decay=cfg.weight_decay
        )
    except AttributeError:                       # older TF without AdamW
        optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)

    state = TrainState(encoder, optimizer)
    latest_mgr, best_mgr = _managers(state, cfg)

    if cfg.resume and latest_mgr.latest_checkpoint:
        state.checkpoint.restore(latest_mgr.latest_checkpoint).expect_partial()
        print(f"[ckpt] resumed from {latest_mgr.latest_checkpoint} "
              f"at step {int(state.step)} / epoch {int(state.epoch)} "
              f"(best {cfg.best_metric}={float(state.best_metric):.4f})")
    else:
        print("[ckpt] starting a fresh run")

    train_writer = tf.summary.create_file_writer(os.path.join(cfg.tb_dir, "train"))
    val_writer = tf.summary.create_file_writer(os.path.join(cfg.tb_dir, "val"))

    query_is_per_record = is_per_record(cfg.query_instruction)
    q_instr = None if query_is_per_record else get_instruction(cfg.query_instruction)
    d_instr = get_instruction(cfg.doc_instruction)

    with train_writer.as_default():
        tf.summary.text("config", json.dumps(cfg.to_hparams(), indent=2), step=0)
        tf.summary.text("instruction/query", describe(cfg.query_instruction), step=0)
        tf.summary.text("instruction/doc", describe(cfg.doc_instruction), step=0)
        tf.summary.text("data/noise_summary",
                        json.dumps({k: round(v, 4)
                                    for k, v in data.noise_summary.items()},
                                   indent=2), step=0)
        train_writer.flush()

    step_fn = _make_train_step(encoder, optimizer, cfg)
    rng = np.random.default_rng(cfg.seed)
    start_epoch = int(state.epoch)
    history: List[Dict] = []
    epochs_without_improvement = 0
    stopped_early = False

    print(f"[train] {n_train} pairs | {steps_per_epoch} steps/epoch | "
          f"{cfg.epochs} epochs | {total_steps} total steps")
    print(f"[train] query instruction -> {describe(cfg.query_instruction)}")
    print(f"[train] doc   instruction -> {describe(cfg.doc_instruction)}")
    if cfg.early_stopping_patience > 0:
        print(f"[train] early stopping: patience={cfg.early_stopping_patience} "
              f"evals, min_delta={cfg.early_stopping_min_delta} on {cfg.best_metric}")

    for epoch in range(start_epoch, cfg.epochs):
        order = rng.permutation(n_train)
        epoch_start = time.time()
        running = []

        for b in range(steps_per_epoch):
            idx = order[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            if len(idx) < 2:                     # in-batch negatives need >= 2
                continue

            batch_clean = [clean_train[i] for i in idx]
            if query_is_per_record:
                batch_q_instr = data.per_record_instructions(
                    [int(train_row_ids[i]) for i in idx],
                    shuffled=(cfg.query_instruction == "record_shuffled"),
                )
            else:
                batch_q_instr = q_instr
            q_batch = encoder.tokenize([dirty_train[i] for i in idx], batch_q_instr)
            d_batch = encoder.tokenize(batch_clean, d_instr)
            dup_mask = tf.convert_to_tensor(_duplicate_mask(batch_clean))
            stats = step_fn(q_batch, d_batch, dup_mask)

            state.step.assign_add(1)
            gstep = int(state.step)
            running.append(float(stats["loss"]))

            with train_writer.as_default():
                tf.summary.scalar("train/loss", float(stats["loss"]), step=gstep)
                tf.summary.scalar("train/pos_sim", float(stats["pos_sim"]), step=gstep)
                tf.summary.scalar("train/neg_sim", float(stats["neg_sim"]), step=gstep)
                tf.summary.scalar("train/margin", float(stats["margin"]), step=gstep)
                tf.summary.scalar("train/in_batch_accuracy",
                                  float(stats["in_batch_accuracy"]), step=gstep)
                tf.summary.scalar("train/grad_norm", float(stats["grad_norm"]), step=gstep)
                tf.summary.scalar("train/learning_rate",
                                  float(schedule(tf.cast(gstep, tf.float32))),
                                  step=gstep)

            if gstep % 20 == 0:
                print(f"  epoch {epoch + 1}/{cfg.epochs} "
                      f"step {gstep}/{total_steps} "
                      f"loss {np.mean(running[-20:]):.4f} "
                      f"margin {float(stats['margin']):.4f}", flush=True)

            # ---- periodic validation + best-checkpoint tracking ----------
            if cfg.eval_every_steps and gstep % cfg.eval_every_steps == 0:
                res = evaluate(encoder, data, cfg, split="val",
                               writer=val_writer, step=gstep,
                               log_visuals=False, save_results=False,
                               verbose=True)
                score = float(res["retrieval"].get(
                    cfg.best_metric, res["retrieval"]["recall@1"]))
                history.append({"step": gstep, cfg.best_metric: score})
                if score > float(state.best_metric) + cfg.early_stopping_min_delta:
                    state.best_metric.assign(score)
                    saved = best_mgr.save()
                    print(f"  [ckpt] new best {cfg.best_metric}={score:.4f} "
                          f"-> {saved}")
                    with val_writer.as_default():
                        tf.summary.scalar("val/best_so_far", score, step=gstep)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                latest_mgr.save()

                if (cfg.early_stopping_patience > 0
                        and epochs_without_improvement >= cfg.early_stopping_patience):
                    print(f"  [early-stop] no improvement in {cfg.best_metric} for "
                          f"{epochs_without_improvement} evals (patience="
                          f"{cfg.early_stopping_patience}) - stopping at step {gstep}")
                    stopped_early = True
                    break

        state.epoch.assign(epoch + 1)
        latest_mgr.save()
        print(f"[train] epoch {epoch + 1} done in "
              f"{time.time() - epoch_start:.1f}s, "
              f"mean loss {np.mean(running) if running else float('nan'):.4f}")
        if stopped_early:
            break

    # ---- end-of-training validation ------------------------------------
    final_val = evaluate(encoder, data, cfg, split="val",
                         writer=val_writer, step=int(state.step),
                         log_visuals=False, save_results=False)
    final_score = float(final_val["retrieval"].get(
        cfg.best_metric, final_val["retrieval"]["recall@1"]))
    if final_score > float(state.best_metric):
        state.best_metric.assign(final_score)
        best_mgr.save()
        print(f"[ckpt] new best {cfg.best_metric}={final_score:.4f} at end of training")
    latest_mgr.save()

    with open(os.path.join(cfg.results_dir, "train_history.json"),
              "w", encoding="utf-8") as f:
        json.dump({"history": history,
                   "best_metric": float(state.best_metric),
                   "final_step": int(state.step)}, f, indent=2)

    return {"best_metric": float(state.best_metric), "steps": int(state.step)}


# --------------------------------------------------------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Fine-tune an instruction-conditioned encoder on AddressTable"
    )
    ap.add_argument("--run", default=None, help="run name (folder under runs/)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore existing checkpoints and start fresh")
    ap.add_argument("--eval-after", action="store_true",
                    help="evaluate the best checkpoint on test when finished")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args()

    cfg = ExperimentConfig()
    if args.run:
        cfg.run_name = args.run
    if args.overrides:
        cfg = parse_overrides(args.overrides, cfg)
    if args.no_resume:
        cfg.resume = False

    print(f"[run] {cfg.run_name}  (config fingerprint {cfg.fingerprint})")
    data = load_dataset(cfg)
    encoder = build_encoder(cfg, data)

    if cfg.do_train:
        train(cfg, data, encoder)

    if args.eval_after or not cfg.do_train:
        from evaluate import evaluate, pick_threshold_on_val
        restore_encoder(encoder, cfg, prefer_best=True)
        writer = tf.summary.create_file_writer(os.path.join(cfg.tb_dir, "test"))
        frozen_threshold = pick_threshold_on_val(encoder, data, cfg)
        evaluate(encoder, data, cfg, split="test", writer=writer,
                 step=0, log_visuals=True, save_results=True,
                 frozen_threshold=frozen_threshold)

    print(f"\nTensorBoard:  tensorboard --logdir \"{cfg.run_dir}\"")


if __name__ == "__main__":
    main()
