#!/usr/bin/env python3
"""
V4 Binary Clip Training — VideoMAEv2 Data-Centric Pipeline
With Discord webhook notifications for remote monitoring.
"""

import os
import sys
import argparse
import time
import traceback
import urllib.request
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from dataset_binary_clips import BinaryClipDataset
from model_binary import BinaryAccidentModel

# ============================================================================
# Discord Webhook
# ============================================================================

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1486599758792491028/Z3pZhS0kQpqpD6vjPNvP32Y8vrXu5jT5GPI3PA8OmemGHMjgncOmSuxIO7WKD7M_6_8b"

def send_discord_msg(content):
    try:
        req = urllib.request.Request(
            DISCORD_WEBHOOK_URL + "?wait=true",
            data=json.dumps({"content": content}).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8")).get("id")
    except Exception:
        pass
    return None

def edit_discord_msg(msg_id, content):
    if not msg_id:
        return
    try:
        req = urllib.request.Request(
            f"{DISCORD_WEBHOOK_URL}/messages/{msg_id}",
            data=json.dumps({"content": content}).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
            method="PATCH"
        )
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception:
        pass

# ============================================================================
# Utilities
# ============================================================================

def log(msg):
    """Flush-safe print for SLURM log files."""
    print(msg, flush=True)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_csv", type=str, default="/ihub/homedirs/sc_hrrs/arkya/sim_dataset/binary_clips_metadata.csv")
    parser.add_argument("--clips_dir", type=str, default="/ihub/homedirs/sc_hrrs/arkya/sim_dataset")
    parser.add_argument("--nexar_weights", type=str, default="/ihub/homedirs/sc_hrrs/arkya/accident_competetion_graph/nexar-solution-main/weights/best.pth")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--samples_per_epoch", type=int, default=4000)
    parser.add_argument("--freeze_backbone", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Debug: Environment ──
    log("=" * 60)
    log(f"Device: {device}")
    if device.type == 'cuda':
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log(f"Args: {vars(args)}")
    log("=" * 60)

    send_discord_msg(
        f"🚀 **V4 Binary Training Started**\n"
        f"Device: `{device}` | BS: `{args.batch_size}` | LR: `{args.lr}` | Epochs: `{args.epochs}`"
    )

    # ── Load & Split ──
    log("Loading clips metadata CSV...")
    df = pd.read_csv(args.clips_csv)
    log(f"Total clips in CSV: {len(df)}")
    log(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    unique_videos = df['video_name'].unique()
    log(f"Unique videos: {len(unique_videos)}")

    # 90/10 video-level split (deterministic)
    val_cut = max(1, int(len(unique_videos) * 0.1))
    val_vids = unique_videos[:val_cut]
    train_vids = unique_videos[val_cut:]
    log(f"Train videos: {len(train_vids)} | Val videos: {len(val_vids)}")

    train_dataset = BinaryClipDataset(args.clips_csv, args.clips_dir, mode="train", apply_degradation=True, valid_video_names=train_vids)
    val_dataset = BinaryClipDataset(args.clips_csv, args.clips_dir, mode="val", apply_degradation=False, valid_video_names=val_vids)

    log(f"Train Clips: {len(train_dataset)} | Val Clips: {len(val_dataset)}")

    # ── Debug: Class balance in train set ──
    train_labels = [s['label'] for s in train_dataset.samples]
    train_pos = sum(train_labels)
    train_neg = len(train_labels) - train_pos
    log(f"Train Label Balance → Positive: {train_pos} | Negative: {train_neg} | Ratio: {train_pos/(train_neg+1e-9):.3f}")

    val_labels = [s['label'] for s in val_dataset.samples]
    val_pos = sum(val_labels)
    val_neg = len(val_labels) - val_pos
    log(f"Val   Label Balance → Positive: {val_pos} | Negative: {val_neg} | Ratio: {val_pos/(val_neg+1e-9):.3f}")

    # WeightedRandomSampler needs DoubleTensor
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(train_dataset.sample_weights).double(),
        num_samples=args.samples_per_epoch,
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    log(f"Train batches/epoch: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ──
    log("Initializing Model...")
    model = BinaryAccidentModel(nexar_weights_path=args.nexar_weights, freeze_backbone=args.freeze_backbone)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # ────────────────────────────────────────────────────────
        # TRAINING
        # ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        batch_pos_count = 0
        batch_neg_count = 0

        total_steps = len(train_loader)
        train_discord_id = send_discord_msg(
            f"⏳ **Epoch {epoch+1}/{args.epochs} Training** — starting..."
        )

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            scene_idx = batch["scene_idx"].to(device, non_blocking=True)
            weather_idx = batch["weather_idx"].to(device, non_blocking=True)
            day_time_idx = batch["day_time_idx"].to(device, non_blocking=True)

            # Debug: first batch shape check
            if epoch == 0 and step == 0:
                log(f"[DEBUG] First batch pixel_values shape: {pixel_values.shape}")
                log(f"[DEBUG] First batch labels: {labels.tolist()}")
                log(f"[DEBUG] First batch pixel_values range: [{pixel_values.min():.3f}, {pixel_values.max():.3f}]")

            with autocast('cuda'):
                logits = model(pixel_values, scene_idx, weather_idx, day_time_idx)
                loss = criterion(logits, labels)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == total_steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * args.grad_accum
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            # Track per-batch class distribution from sampler
            batch_pos_count += (labels == 1).sum().item()
            batch_neg_count += (labels == 0).sum().item()

            pbar.set_postfix({"Loss": f"{loss.item()*args.grad_accum:.4f}", "Acc": f"{correct_train/total_train:.4f}"})

            # ── Discord Progress: update every 10% ──
            if (step + 1) % max(1, total_steps // 10) == 0 or (step + 1) == total_steps:
                pct = int(100 * (step + 1) / total_steps)
                filled = int(pct / 10)
                bar = "█" * filled + "▒" * (10 - filled)
                cur_loss = train_loss / (step + 1)
                cur_acc = correct_train / max(total_train, 1)
                prog_txt = (
                    f"⏳ **Epoch {epoch+1}/{args.epochs} Training**\n"
                    f"Progress: `[{bar}] {pct}%` ({step+1}/{total_steps})\n"
                    f"Loss: `{cur_loss:.4f}` | Acc: `{cur_acc:.4f}` | Pos/Neg: `{batch_pos_count}/{batch_neg_count}`"
                )
                edit_discord_msg(train_discord_id, prog_txt)

        train_time = time.time() - epoch_start
        avg_train_loss = train_loss / max(len(train_loader), 1)
        train_acc = correct_train / max(total_train, 1)

        log(f"\n{'='*60}")
        log(f"Epoch {epoch+1}/{args.epochs} Train Summary:")
        log(f"  Time: {train_time:.1f}s | Avg Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f}")
        log(f"  Sampler Class Balance → Pos sampled: {batch_pos_count} | Neg sampled: {batch_neg_count}")
        if device.type == 'cuda':
            log(f"  GPU Mem: {torch.cuda.memory_allocated()/1e9:.2f}GB used / {torch.cuda.max_memory_allocated()/1e9:.2f}GB peak")

        scheduler.step()

        # ────────────────────────────────────────────────────────
        # VALIDATION
        # ────────────────────────────────────────────────────────
        val_start = time.time()
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_preds = []
        all_probs = []

        val_total_steps = len(val_loader)
        val_discord_id = send_discord_msg(
            f"⏳ **Epoch {epoch+1}/{args.epochs} Validation** — starting..."
        )

        with torch.no_grad():
            for val_step, batch in enumerate(tqdm(val_loader, desc="[Val]")):
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                scene_idx = batch["scene_idx"].to(device, non_blocking=True)
                weather_idx = batch["weather_idx"].to(device, non_blocking=True)
                day_time_idx = batch["day_time_idx"].to(device, non_blocking=True)

                with autocast('cuda'):
                    logits = model(pixel_values, scene_idx, weather_idx, day_time_idx)
                    v_loss = criterion(logits, labels)

                val_loss += v_loss.item() * labels.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # ── Discord Progress: update every 10% ──
                if (val_step + 1) % max(1, val_total_steps // 10) == 0 or (val_step + 1) == val_total_steps:
                    pct = int(100 * (val_step + 1) / val_total_steps)
                    filled = int(pct / 10)
                    bar = "█" * filled + "▒" * (10 - filled)
                    cur_val_loss = val_loss / max(total_val, 1)
                    cur_val_acc = correct_val / max(total_val, 1)
                    val_prog = (
                        f"⏳ **Epoch {epoch+1}/{args.epochs} Validation**\n"
                        f"Progress: `[{bar}] {pct}%` ({val_step+1}/{val_total_steps})\n"
                        f"Loss: `{cur_val_loss:.4f}` | Acc: `{cur_val_acc:.4f}`"
                    )
                    edit_discord_msg(val_discord_id, val_prog)

        val_time = time.time() - val_start
        val_loss /= max(total_val, 1)
        val_acc = correct_val / max(total_val, 1)

        # Confusion Matrix and Classification Report
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=["Negative", "Positive"], zero_division=0)

        log(f"\nEpoch {epoch+1} Validation Summary:")
        log(f"  Time: {val_time:.1f}s | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        log(f"  Prob stats → min: {min(all_probs):.4f} | max: {max(all_probs):.4f} | mean: {np.mean(all_probs):.4f}")
        log(f"  Confusion Matrix:\n{cm}")
        log(f"  Classification Report:\n{report}")
        log(f"{'='*60}\n")

        # ── Discord: Epoch Summary ──
        saved_tag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.output_dir, "best_binary_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            log(f"  --> Saved new best model (val_loss={val_loss:.4f}) to {ckpt_path}")
            saved_tag = " 💾 **New Best Saved!**"

        epoch_summary = (
            f"✅ **Epoch {epoch+1}/{args.epochs} Completed**{saved_tag}\n"
            f"⏱️ Train: `{train_time:.0f}s` | Val: `{val_time:.0f}s`\n"
            f"📉 Train Loss: `{avg_train_loss:.4f}` | Val Loss: `{val_loss:.4f}`\n"
            f"🎯 Train Acc: `{train_acc:.4f}` | Val Acc: `{val_acc:.4f}`\n"
            f"📊 Sampler Pos/Neg: `{batch_pos_count}/{batch_neg_count}`\n"
            f"🔢 Confusion: TN=`{cm[0][0]}` FP=`{cm[0][1]}` FN=`{cm[1][0]}` TP=`{cm[1][1]}`\n"
            f"📈 Prob range: `[{min(all_probs):.3f}, {max(all_probs):.3f}]` mean=`{np.mean(all_probs):.3f}`"
        )
        send_discord_msg(epoch_summary)

    send_discord_msg("🎉 **V4 Binary Training Finished Successfully!**")
    log("Training complete!")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        err = traceback.format_exc()
        log(f"FATAL ERROR:\n{err}")
        # Discord limit is 2000 chars
        send_discord_msg(f"🚨 **V4 TRAINING CRASHED** 🚨\n```python\n{err[-1800:]}\n```")
        raise e
