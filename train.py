"""
Training script for ViT-S/16 on ImageNet-1k with DDP.
"""

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam

from data import get_dataloaders
from model import vit_small_patch16_224


def setup_distributed():
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
        torch.cuda.set_device(0)
    return rank, world_size, local_rank


def cosine_scheduler(base_lr, final_lr, total_steps, warm_steps=0):
    warm_schedule = np.array([])
    if warm_steps > 0:
        warm_schedule = np.linspace(0, base_lr, warm_steps)
    iters = np.arange(total_steps - warm_steps)
    schedule = final_lr + 0.5 * (base_lr - final_lr) * (
        1 + np.cos(np.pi * iters / len(iters))
    )
    schedule = np.concatenate((warm_schedule, schedule))
    assert len(schedule) == total_steps
    return schedule


def mixup(x, y, n_classes, p=0.2):
    """https://github.com/google-research/big_vision/blob/main/big_vision/utils.py#L1146"""
    a = np.random.beta(p, p)
    a = max(a, 1 - a)  # ensure a >= 0.5 so that `unrolled x` is dominant
    mixed_x = a * x + (1 - a) * x.roll(1, dims=0)
    y_onehot = torch.zeros(y.size(0), n_classes, device=y.device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1)  # one-hot encoding
    mixed_y = a * y_onehot + (1 - a) * y_onehot.roll(1, dims=0)
    return mixed_x, mixed_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--mom", type=float, default=0.9)
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--data", type=str, default="i1k")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--warm_ratio", type=float, default=0.1)
    parser.add_argument("--mixup_p", type=float, default=0.2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--d_embed", type=int, default=384)
    parser.add_argument("--dir_output", type=str, required=True)
    parser.add_argument("--dir_data", type=str, required=True)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True
    if rank == 0:
        os.makedirs(args.dir_output, exist_ok=True)
        wandb.init(project="adam-sgd-gap", mode="disabled" if args.debug else "online")
        wandb.config.update(args)
        wandb.config.update({"bs": args.bs * args.accum_steps}, allow_val_change=True)

    tr_loader, vl_loader, n_classes, steps_per_epoch = get_dataloaders(
        dataset=args.data,
        dir_data=args.dir_data,
        batch_size=args.bs,
        n_workers=args.n_workers,
        aug=args.aug,
    )

    model = vit_small_patch16_224(
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_embed=args.d_embed,
        n_classes=n_classes,
    ).cuda()
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if args.compile:
        model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()

    total_steps = args.epochs * steps_per_epoch // args.accum_steps
    if args.opt == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.mom)
    elif args.opt == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.mom, args.mom))
    else:
        raise ValueError(f"Unknown optimizer: {args.opt}")
    scheduler = cosine_scheduler(
        base_lr=args.lr,
        final_lr=0,
        total_steps=total_steps,
        warm_steps=int(args.warm_ratio * total_steps),
    )

    best_acc = 0.0
    global_step = 0
    for epoch in range(args.epochs):
        # Train
        model.train()
        for step, (x, y) in enumerate(tr_loader):
            if step >= steps_per_epoch:
                break

            lr = scheduler[global_step]
            for p in optimizer.param_groups:
                p["lr"] = lr

            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            x, y_soft = mixup(x, y, n_classes, p=args.mixup_p)
            with autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = (
                    -torch.sum(y_soft * torch.log_softmax(logits, dim=1), dim=1).mean()
                    / args.accum_steps
                )
            loss.backward()

            if (step + 1) % args.accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if rank == 0 and global_step % args.log_interval == 0:
                tr_loss = loss.item() * args.accum_steps
                print(f"step {global_step} tr_loss {tr_loss:.4f} lr {lr:.6f}")
                wandb.log({"train/loss": tr_loss, "train/lr": lr})

            # Evaluate
            if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                model.eval()
                vl_loss, vl_correct1, vl_correct5, vl_n = 0, 0, 0, 0
                with torch.no_grad():
                    for x, y in vl_loader:
                        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                        with autocast("cuda", dtype=torch.bfloat16):
                            out = model(x)
                        vl_loss += criterion(out, y).item() * x.size(0)
                        top5 = out.topk(5, dim=1)[1]
                        vl_correct1 += top5[:, 0].eq(y).sum().item()
                        vl_correct5 += top5.eq(y.view(-1, 1)).sum().item()
                        vl_n += x.size(0)
                metrics = [vl_loss, vl_correct1, vl_correct5, vl_n]
                metrics = torch.tensor(metrics, device="cuda")
                if world_size > 1:
                    dist.all_reduce(metrics)
                vl_loss, vl_acc1, vl_acc5 = (
                    metrics[0].item() / metrics[3].item(),
                    metrics[1].item() / metrics[3].item() * 100,
                    metrics[2].item() / metrics[3].item() * 100,
                )
                if rank == 0:
                    print(f"step {global_step} vl_acc1 {vl_acc1:.2f}")
                    wandb.log(
                        {"val/loss": vl_loss, "val/acc1": vl_acc1, "val/acc5": vl_acc5}
                    )
                    w = (
                        model.module.state_dict()
                        if world_size > 1
                        else model.state_dict()
                    )
                    torch.save(w, os.path.join(args.dir_output, "last.pth"))
                    if vl_acc1 > best_acc:
                        best_acc = vl_acc1
                        torch.save(w, os.path.join(args.dir_output, "best.pth"))
                model.train()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
