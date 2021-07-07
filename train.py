#!/usr/bin/env python3
"""Train S2VC model."""

import argparse
import datetime
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from data import IntraSpeakerDataset, collate_batch, plot_attn
from models import S2VC, get_cosine_schedule_with_warmup

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--total_steps", type=int, default=250000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--valid_steps", type=int, default=1000)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--accu_steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument('-s', "--src_feat", type=str, default='cpc')
    parser.add_argument('-r', "--ref_feat", type=str, default='cpc')
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--lr_reduction", action="store_true")
    parser.add_argument("--comment", type=str)


    return vars(parser.parse_args())


def model_fn(batch, model, criterion, device):
    """Forward a batch through model."""

    srcs, src_masks, tgts, tgt_masks, tgt_mels, overlap_lens = batch

    srcs = srcs.to(device)
    src_masks = src_masks.to(device)
    tgts = tgts.to(device)
    tgt_masks = tgt_masks.to(device)
    tgt_mels = tgt_mels.to(device)

    refs = tgts
    ref_masks = tgt_masks

    outs, attns = model(srcs, refs, src_masks=src_masks, ref_masks=ref_masks)
            
    losses = []
    for out, tgt_mel, attn, overlap_len in zip(outs.unbind(), tgt_mels.unbind(), attns[-1], overlap_lens):
        loss = criterion(out[:, :overlap_len], tgt_mel[:, :overlap_len])
        losses.append(loss)
    try:
        attns_plot = []
        for i in range(len(attns)):
            attns_plot.append(attns[i][0][:overlap_lens[0], :overlap_lens[0]])
    except:
        pass

        
    return sum(losses) / len(losses), attns_plot


def valid(dataloader, model, criterion, device):
    """Validate on validation set."""

    model.eval()
    running_loss = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, attns = model_fn(batch, model, criterion, device)
            running_loss += loss.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(loss=f"{running_loss / (i+1):.2f}")

    pbar.close()
    model.train()

    return running_loss / len(dataloader), attns


def main(
    data_dir,
    save_dir,
    total_steps,
    warmup_steps,
    valid_steps,
    log_steps,
    save_steps,
    n_samples,
    accu_steps,
    batch_size,
    n_workers,
    src_feat,
    ref_feat,
    preload,
    lr_reduction,
    comment,
):
    """Main function."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata_path = Path(data_dir) / "metadata.json"
    
    dataset = IntraSpeakerDataset(
        data_dir, metadata_path, src_feat, ref_feat, n_samples, preload
    )
    input_dim, ref_dim, tgt_dim = dataset.get_feat_dim()
    lengths = [trainlen := int(0.9 * len(dataset)), len(dataset) - trainlen]
    trainset, validset = random_split(dataset, lengths)
    print(f'Input dim: {input_dim}, Reference dim: {ref_dim}, Target dim: {tgt_dim}')
    model = S2VC(input_dim, ref_dim).to(device)
    model = torch.jit.script(model)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size * accu_steps,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        # shuffle to make the plot on tensorboard differenct
        shuffle=True,
        collate_fn=collate_batch,
    )
    train_iterator = iter(train_loader)

    if comment is not None:
        log_dir = "logs/"
        log_dir += datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        log_dir += "_" + comment
        writer = SummaryWriter(log_dir)

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)


    learning_rate = 5e-5
    criterion = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_loss = float("inf")
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        if step == 40002:
            file = open('completed.txt', 'a')
            print(f'{comment} completed', file=file)
            break
        batch_loss = 0.0

        for _ in range(accu_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            loss, attns = model_fn(batch, model, criterion, device)
            loss = loss / accu_steps
            batch_loss += loss.item()
            loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        pbar.update()
        pbar.set_postfix(loss=f"{batch_loss:.2f}", step=step + 1)

        if step % log_steps == 0 and comment is not None:
            writer.add_scalar("Loss/train", batch_loss, step)
            try:
                attn = [attns[i].unsqueeze(0) for i in range(len(attns))]
                figure = plot_attn(attn, save=False)
                writer.add_figure(f"Image/Train-Attentions.png", figure, step + 1)
            except:
                pass

        if (step + 1) % valid_steps == 0:
            pbar.close()

            valid_loss, attns = valid(valid_loader, model, criterion, device)

            if comment is not None:
                writer.add_scalar("Loss/valid", valid_loss, step + 1)
                try:
                    attn = [attns[i].unsqueeze(0) for i in range(len(attns))]
                    figure = plot_attn(attn, save=False)
                    writer.add_figure(f"Image/Valid-Attentions.png", figure, step + 1)
                except:
                    pass

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            loss_str = f"{best_loss:.4f}".replace(".", "dot")
            best_ckpt_name = f"retriever-best-loss{loss_str}.pt"

            loss_str = f"{valid_loss:.4f}".replace(".", "dot")
            curr_ckpt_name = f"retriever-step{step+1}-loss{loss_str}.pt"

            current_state_dict = model.state_dict()
            model.cpu()

            model.load_state_dict(best_state_dict)
            model.save(str(save_dir_path / best_ckpt_name))

            model.load_state_dict(current_state_dict)
            model.save(str(save_dir_path / curr_ckpt_name))

            model.to(device)
            pbar.write(f"Step {step + 1}, best model saved. (loss={best_loss:.4f})")

        
    pbar.close()


if __name__ == "__main__":
    main(**parse_args())
