import argparse
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import optuna
import logging
import sys

from tgm import DGBatch, DGraph, TimeDeltaDG
from tgm.data import DGData, DGDataLoader, TemporalSplit
from tgm.util.seed import seed_everything


def _build_bipartite_id_maps(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    user_map: Dict[str, int] = {}
    item_map: Dict[str, int] = {}

    def map_user(x: str) -> int:
        if x not in user_map:
            user_map[x] = len(user_map)
        return user_map[x]

    def map_item(x: str) -> int:
        if x not in item_map:
            item_map[x] = len(item_map)
        return item_map[x]

    df = df.copy()
    df["from"] = df["from"].astype(str).map(map_user).astype("int64")
    df["to"] = df["to"].astype(str).map(map_item).astype("int64")
    # shift items after users
    item_offset = int(len(user_map))
    df["to"] = (df["to"] + item_offset).astype("int64")
    return df, user_map, item_map


def bounds_event_ratio_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    time_col: str = "timestamp",
) -> tuple[int, int]:
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    times = df_sorted[time_col].to_numpy()
    n = int(times.shape[0])

    idx_val = int(n * float(train_ratio))
    idx_test = int(n * float(train_ratio + val_ratio))

    val_time = int(times[idx_val])
    test_time = int(times[idx_test])
    return val_time, test_time


class GCNEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_hops: int,
        dropout: float,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        if num_hops == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
            for _ in range(num_hops - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
            self.convs.append(GCNConv(hidden_dim, out_channels))

    def forward(self, batch: DGBatch, node_feat: torch.Tensor) -> torch.Tensor:
        edge_index = torch.stack([batch.edge_src, batch.edge_dst], dim=0)
        x = node_feat
        if len(self.convs) == 1:
            return self.convs[0](x, edge_index)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class DotProductDecoder(torch.nn.Module):
    def forward(self, z_users: torch.Tensor, z_items: torch.Tensor) -> torch.Tensor:
        return torch.matmul(z_users, z_items.t())


def _select_last_event_per_user(
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_time: torch.Tensor,
    item_offset: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """inside one time-bin, take one positive per user: the last by timestamp"""
    u = edge_src
    t = edge_time
    big = int(t.max().item()) + 1
    key = u.to(torch.int64) * big + t.to(torch.int64)
    order = torch.argsort(key)  # increasing (u, t)
    u_sorted = u[order]

    is_last = torch.empty_like(u_sorted, dtype=torch.bool)
    is_last[:-1] = u_sorted[:-1] != u_sorted[1:]
    is_last[-1] = True

    picked = order[is_last]
    users_global = edge_src[picked]
    pos_items_local = (edge_dst[picked] - int(item_offset)).to(torch.long)
    return users_global, pos_items_local


def _update_window_prefix(
    window: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> DGBatch:
    """concatenate snapshot edges in the window into one prefix batch"""
    if len(window) == 0:
        edge_src = torch.empty(0, dtype=torch.long, device=device)
        edge_dst = torch.empty(0, dtype=torch.long, device=device)
        edge_time = torch.empty(0, dtype=torch.long, device=device)
        return DGBatch(edge_src=edge_src, edge_dst=edge_dst, edge_time=edge_time)
    # already on device
    src = torch.cat([x[0] for x in window], dim=0)
    dst = torch.cat([x[1] for x in window], dim=0)
    tim = torch.cat([x[2] for x in window], dim=0)
    return DGBatch(edge_src=src, edge_dst=dst, edge_time=tim)


def train_epoch(
    loader: DGDataLoader,
    snapshots_loader: DGDataLoader,
    encoder: nn.Module,
    decoder: nn.Module,
    node_emb: nn.Embedding,
    opt: torch.optim.Optimizer,
    conversion_rate: int,
    num_items: int,
    item_offset: int,
    window_snapshots: int,
) -> float:
    encoder.train()
    decoder.train()
    node_emb.train()

    device = loader.dgraph.device
    node_feat = node_emb.weight

    item_ids_global = torch.arange(
        int(item_offset), int(item_offset + num_items),
        device=device, dtype=torch.long
    )

    snapshots_iter = iter(snapshots_loader)
    next_snapshot = None

    W = int(window_snapshots)
    window: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(
        maxlen=max(W, 0)
    )  # (edge_src, edge_dst, edge_time)

    prefix_batch = _update_window_prefix(deque(), device=device)
    z = encoder(prefix_batch, node_feat)
    prev_prefix_sig = (0, -1)

    total_loss = 0.0
    total_users = 0

    for batch in loader:
        batch_start = int(batch.edge_time.min().item())  # raw time (seconds)

        # Advance snapshots strictly BEFORE this batch
        while True:
            if next_snapshot is None:
                try:
                    next_snapshot = next(snapshots_iter)
                except StopIteration:
                    break

            snap_idx = int(next_snapshot.edge_time[-1].item())  # время в дискретизированных единицах = snapshot sid
            snap_end = (snap_idx + 1) * int(conversion_rate)    # end time in raw units

            if snap_end <= batch_start:
                if W > 0:
                    window.append((next_snapshot.edge_src, next_snapshot.edge_dst, next_snapshot.edge_time))
                next_snapshot = None
            else:
                break

        last_sid = int(window[-1][2][-1].item()) if len(window) > 0 else -1
        sig = (len(window), last_sid)
        if sig != prev_prefix_sig:
            prefix_batch = _update_window_prefix(window, device=device)
            z = encoder(prefix_batch, node_feat)
            prev_prefix_sig = sig

        # one target per user
        users_global, pos_items_local = _select_last_event_per_user(
            batch.edge_src, batch.edge_dst, batch.edge_time, item_offset
        )

        users_z = z[users_global]        # [U, d]
        items_z = z[item_ids_global]     # [I, d]
        logits = decoder(users_z, items_z)  # [U, I]

        loss = F.cross_entropy(logits, pos_items_local)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        U = int(users_global.numel())
        total_loss += float(loss.item()) * U
        total_users += U

    return float(total_loss / max(total_users, 1))


@torch.no_grad()
def eval_metrics(
    loader: DGDataLoader,
    snapshots_loader: DGDataLoader,
    encoder: nn.Module,
    decoder: nn.Module,
    node_emb: nn.Embedding,
    conversion_rate: int,
    num_items: int,
    item_offset: int,
    k: int,
    window_snapshots: int,
) -> Dict[str, float]:
    encoder.eval()
    decoder.eval()
    node_emb.eval()

    device = loader.dgraph.device
    node_feat = node_emb.weight

    k_eff = int(k)
    num_users = int(item_offset)

    item_ids_global = torch.arange(
        int(item_offset), int(item_offset + num_items),
        device=device, dtype=torch.long
    )

    ndcg_sum = torch.zeros(num_users, device=device, dtype=torch.float32)
    ndcg_cnt = torch.zeros(num_users, device=device, dtype=torch.float32)

    last_topk = torch.full((num_users, k_eff), -1, device=device, dtype=torch.long)

    snapshots_iter = iter(snapshots_loader)
    next_snapshot = None

    W = int(window_snapshots)
    window: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(maxlen=max(W, 0))

    prefix_batch = _update_window_prefix(deque(), device=device)
    z = encoder(prefix_batch, node_feat)
    prev_prefix_sig = (0, -1)

    for batch in loader:
        batch_start = int(batch.edge_time.min().item())

        while True:
            if next_snapshot is None:
                try:
                    next_snapshot = next(snapshots_iter)
                except StopIteration:
                    break

            snap_idx = int(next_snapshot.edge_time[-1].item())
            snap_end = (snap_idx + 1) * int(conversion_rate)
            if snap_end <= batch_start:
                if W > 0:
                    window.append((next_snapshot.edge_src, next_snapshot.edge_dst, next_snapshot.edge_time))
                next_snapshot = None
            else:
                break

        last_sid = int(window[-1][2][-1].item()) if len(window) > 0 else -1
        sig = (len(window), last_sid)
        if sig != prev_prefix_sig:
            prefix_batch = _update_window_prefix(window, device=device)
            z = encoder(prefix_batch, node_feat)
            prev_prefix_sig = sig

        users_global, pos_items_local = _select_last_event_per_user(
            batch.edge_src, batch.edge_dst, batch.edge_time, item_offset
        )

        users_z = z[users_global]            # [U, d]
        items_z = z[item_ids_global]         # [I, d]
        logits = decoder(users_z, items_z)   # [U, I]

        topk_idx = torch.topk(logits, k=k_eff, dim=1, largest=True).indices  # [U, k]

        # NDCG@k векторно
        match = (topk_idx == pos_items_local.view(-1, 1))
        hit = match.any(dim=1)  # [U]
        rank = match.float().argmax(dim=1)  # [U], 0..k-1
        ndcg_batch = torch.zeros_like(rank, dtype=torch.float32, device=device)
        ndcg_batch[hit] = 1.0 / torch.log2(rank[hit].to(torch.float32) + 2.0)

        ndcg_sum.scatter_add_(0, users_global, ndcg_batch)
        ndcg_cnt.scatter_add_(0, users_global, torch.ones_like(ndcg_batch))

        last_topk[users_global] = topk_idx

    mask = ndcg_cnt > 0
    if mask.any():
        user_means = ndcg_sum[mask] / ndcg_cnt[mask]
        ndcg = float(user_means.mean().item())
    else:
        ndcg = 0.0

    valid = last_topk.view(-1)
    valid = valid[valid >= 0]
    if valid.numel() == 0:
        coverage = 0.0
    else:
        coverage = float(torch.unique(valid).numel()) / float(max(1, int(num_items)))

    return {"NDCG": ndcg, "Coverage": coverage}


@dataclass
class EarlyStopper:
    patience: int
    key: str = "NDCG"
    min_delta: float = 0.0
    cnt: int = 0
    best: float = -float("inf")
    best_epoch: int = -1
    best_encoder_state: Optional[dict] = None
    best_decoder_state: Optional[dict] = None
    best_node_emb_state: Optional[dict] = None

    def step(self, value: float, encoder: nn.Module, decoder: nn.Module, node_emb: nn.Embedding, epoch: int) -> bool:
        if value > (self.best + self.min_delta):
            self.best = float(value)
            self.best_epoch = int(epoch)
            self.cnt = 0
            # keep best on CPU
            self.best_encoder_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            self.best_decoder_state = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
            self.best_node_emb_state = {k: v.detach().cpu().clone() for k, v in node_emb.state_dict().items()}
            return False

        self.cnt += 1
        return self.cnt >= int(self.patience)

    def restore_best(self, encoder: nn.Module, decoder: nn.Module, node_emb: nn.Embedding) -> None:
        if self.best_encoder_state is not None:
            encoder.load_state_dict(self.best_encoder_state)
        if self.best_decoder_state is not None:
            decoder.load_state_dict(self.best_decoder_state)
        if self.best_node_emb_state is not None:
            node_emb.load_state_dict(self.best_node_emb_state)


def run_train_val(
    *,
    seed: int,
    epochs: int,
    patience: int,
    min_delta: float,
    lr: float,
    dropout: float,
    num_hops: int,
    emb_dim: int,
    window_snapshots: int,
    ndcg_k: int,
    device: torch.device,
    # data-related
    full_num_nodes: int,
    conversion_rate: int,
    num_items: int,
    item_offset: int,
    train_loader: DGDataLoader,
    val_loader: DGDataLoader,
    snapshots_loader: DGDataLoader,
    log_prefix: Optional[str] = None,  # e.g. "trial 3"
) -> dict:
    seed_everything(int(seed))

    node_emb = nn.Embedding(int(full_num_nodes), int(emb_dim)).to(device)
    torch.nn.init.normal_(node_emb.weight, std=0.1)

    encoder = GCNEncoder(
        in_channels=int(emb_dim),
        hidden_dim=int(emb_dim),
        out_channels=int(emb_dim),
        num_hops=int(num_hops),
        dropout=float(dropout),
    ).to(device)

    decoder = DotProductDecoder().to(device)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(node_emb.parameters()),
        lr=float(lr),
    )

    stopper = EarlyStopper(patience=int(patience), min_delta=float(min_delta))

    train_total_t0 = time.perf_counter()
    best_val = -1e18
    best_epoch = 0
    best_val_metrics = {"NDCG": 0.0, "Coverage": 0.0}
    epochs_ran = 0

    for epoch in range(1, int(epochs) + 1):
        epochs_ran = int(epoch)
        loss = train_epoch(
            train_loader,
            snapshots_loader,
            encoder,
            decoder,
            node_emb,
            opt,
            conversion_rate,
            num_items,
            item_offset,
            int(window_snapshots),
        )

        val = eval_metrics(
            val_loader,
            snapshots_loader,
            encoder,
            decoder,
            node_emb,
            conversion_rate,
            num_items,
            item_offset,
            int(ndcg_k),
            int(window_snapshots),
        )

        if float(val["NDCG"]) > best_val:
            best_val = float(val["NDCG"])
            best_epoch = int(epoch)
            best_val_metrics = {"NDCG": float(val["NDCG"]), "Coverage": float(val["Coverage"])}

        if log_prefix is not None:
            print(
                f"{log_prefix} epoch {epoch}: "
                f"loss={float(loss):.6f} "
                f"val_NDCG={float(val['NDCG']):.6f} "
                f"val_Coverage={float(val['Coverage']):.6f}",
                flush=True,
            )

        if stopper.step(float(val["NDCG"]), encoder, decoder, node_emb, epoch):
            break

    stopper.restore_best(encoder, decoder, node_emb)
    train_total_sec = time.perf_counter() - train_total_t0
    avg_epoch_sec = float(train_total_sec) / float(max(int(epochs_ran), 1))

    final_best_epoch = int(stopper.best_epoch if stopper.best_epoch >= 0 else best_epoch)

    return {
        "best_val_ndcg": float(best_val),
        "best_epoch": final_best_epoch,
        "best_val_metrics": dict(best_val_metrics),
        "train_total_sec": float(train_total_sec),
        "avg_epoch_sec": float(avg_epoch_sec),
        "epochs_ran": float(epochs_ran),
        "node_emb": node_emb,
        "encoder": encoder,
        "decoder": decoder,
    }



def main():
    parser = argparse.ArgumentParser(
        description="Temporal bipartite GCN + Optuna",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--num-hops", type=int, default=2, help="GCN hops = number of GCNConv layers")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Same dimension for node_emb and GCN")
    parser.add_argument("--bsize", type=int, default=1, help="DGDataLoader batch_size (in batch_unit units)")

    parser.add_argument(
        "--path-dataset",
        type=str,
        default="data/movielens/ml-1m_ratings_tgm.csv",
        help="CSV with columns: from,to,timestamp (value optional)",
    )
    parser.add_argument("--raw-time-gran", type=str, default="s")
    parser.add_argument(
        "--snapshot-time-gran",
        type=str,
        default="h",
        help="time unit for batching events and snapshots (batch_unit)",
    )
    parser.add_argument("--window-snapshots", type=int, default=24, help="sliding window size in number of snapshots")

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--ndcg-k", type=int, default=20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--log-file-path", type=str, default=None)

    parser.add_argument("--optuna-trials", type=int, default=50, help="If > 0, run Optuna with this number of trials")
    parser.add_argument("--optuna-study-name", type=str, default="tgm_optuna")
    parser.add_argument("--optuna-sampler-seed", type=int, default=1337)

    args = parser.parse_args()

    # Silence all library logs
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log_file_path:
        handlers.append(logging.FileHandler(args.log_file_path))
    logging.basicConfig(level=logging.ERROR, handlers=handlers, format="%(message)s")
    for name in ["tgm", "tgm.data", "tgm.util", "torch_geometric", "optuna"]:
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        optuna.logging.set_verbosity(optuna.logging.ERROR)
    except Exception:
        pass

    seed_everything(args.seed)
    device = torch.device(args.device)

    # data load
    df = pd.read_csv(args.path_dataset)
    df = df.dropna(subset=["from", "to", "timestamp"]).copy()
    df["timestamp"] = df["timestamp"].astype("int64")

    df, user_map, item_map = _build_bipartite_id_maps(df)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # start at 0
    t0 = int(df["timestamp"].min())
    df["timestamp"] = (df["timestamp"] - t0).astype("int64")

    num_users = int(len(user_map))
    num_items = int(len(item_map))
    item_offset = int(num_users)

    full_data = DGData.from_pandas(
        edge_df=df,
        edge_src_col="from",
        edge_dst_col="to",
        edge_time_col="timestamp",
        edge_x_col=None,
        time_delta=args.raw_time_gran,
    )

    val_time_q, test_time_q = bounds_event_ratio_split(df, args.train_ratio, args.val_ratio, time_col="timestamp")
    train_data, val_data, test_data = full_data.split(
        TemporalSplit(val_time=int(val_time_q), test_time=int(test_time_q))
    )

    train_dg = DGraph(train_data, device=device)
    val_dg = DGraph(val_data, device=device)
    test_dg = DGraph(test_data, device=device)

    train_loader = DGDataLoader(train_dg, batch_size=int(args.bsize), batch_unit=args.snapshot_time_gran)
    val_loader = DGDataLoader(val_dg, batch_size=int(args.bsize), batch_unit=args.snapshot_time_gran)
    test_loader = DGDataLoader(test_dg, batch_size=int(args.bsize), batch_unit=args.snapshot_time_gran)

    # message passing graph: make undirected
    df_rev = df.copy()
    df_rev[["from", "to"]] = df_rev[["to", "from"]]
    df_mp = pd.concat([df, df_rev], ignore_index=True)
    df_mp = df_mp.drop_duplicates(subset=["from", "to", "timestamp"]).sort_values("timestamp").reset_index(drop=True)

    all_mp = DGData.from_pandas(
        edge_df=df_mp,
        edge_src_col="from",
        edge_dst_col="to",
        edge_time_col="timestamp",
        edge_x_col=None,
        time_delta=args.raw_time_gran,
    )

    all_data_discretized = all_mp.discretize(args.snapshot_time_gran)
    all_snapshots = DGraph(all_data_discretized, device=device)
    all_snapshots_loader = DGDataLoader(all_snapshots, batch_unit=args.snapshot_time_gran)

    snapshot_td = TimeDeltaDG(args.snapshot_time_gran)
    conversion_rate = int(snapshot_td.convert(train_dg.time_delta))

    emb_dim_fixed = int(args.embedding_dim)

    # Optuna mode
    if int(args.optuna_trials) > 0:
        sampler = optuna.samplers.TPESampler(seed=int(args.optuna_sampler_seed))
        study = optuna.create_study(direction="maximize", study_name=str(args.optuna_study_name), sampler=sampler)

        n_trials = int(args.optuna_trials)

        def objective(trial: optuna.Trial) -> float:
            emb_dim = int(trial.suggest_categorical("embedding_dim", [16, 32, 64, 128]))
            dropout = float(trial.suggest_float("dropout", 0.0, 0.5, step=0.1))
            lr = float(trial.suggest_float("lr", 1e-4, 3e-3, log=True))
            num_hops = int(trial.suggest_categorical("num_hops", [1, 2, 3]))

            trial_prefix = f"[trial {trial.number + 1}/{n_trials}]"
            print(f"\n===== TRIAL {trial.number + 1}/{n_trials} =====", flush=True)
            print(
                f"{trial_prefix} params: "
                f"embedding_dim={emb_dim} num_hops={num_hops} dropout={dropout:.3f} lr={lr:.6g}",
                flush=True,
            )

            out = run_train_val(
                seed=int(args.seed),
                epochs=int(args.epochs),
                patience=int(args.patience),
                min_delta=float(args.min_delta),
                lr=lr,
                dropout=dropout,
                num_hops=num_hops,
                emb_dim=emb_dim,
                window_snapshots=int(args.window_snapshots),
                ndcg_k=int(args.ndcg_k),
                device=device,
                full_num_nodes=int(full_data.num_nodes),
                conversion_rate=int(conversion_rate),
                num_items=int(num_items),
                item_offset=int(item_offset),
                train_loader=train_loader,
                val_loader=val_loader,
                snapshots_loader=all_snapshots_loader,
                log_prefix=trial_prefix,
            )

            bestm = out.get("best_val_metrics", {})
            print(
                f"{trial_prefix} BEST: "
                f"best_epoch={int(out['best_epoch'])} "
                f"best_val_NDCG={float(bestm.get('NDCG', out['best_val_ndcg'])):.6f} "
                f"best_val_Coverage={float(bestm.get('Coverage', 0.0)):.6f} "
                f"train_total_sec={float(out.get('train_total_sec', 0.0)):.2f} "
                f"avg_epoch_sec={float(out.get('avg_epoch_sec', 0.0)):.3f} "
                f"epochs_ran={int(float(out.get('epochs_ran', 0.0)))}",
                flush=True,
            )

            trial.set_user_attr("best_epoch", int(out["best_epoch"]))
            trial.set_user_attr("train_total_sec", float(out["train_total_sec"]))
            return float(out["best_val_ndcg"])

        # optuna version compatibility (show_progress_bar may not exist)
        try:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        except TypeError:
            study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_value = float(study.best_value)

        print("\n===== OPTUNA BEST =====", flush=True)
        print(f"best_value (val NDCG): {best_value:.6f}", flush=True)
        print(f"best_params: {best_params}", flush=True)

        # final run with best params, then test
        out = run_train_val(
            seed=int(args.seed),
            epochs=int(args.epochs),
            patience=int(args.patience),
            min_delta=float(args.min_delta),
            lr=float(best_params["lr"]),
            dropout=float(best_params["dropout"]),
            num_hops=int(best_params["num_hops"]),
            emb_dim=int(best_params["embedding_dim"]),
            window_snapshots=int(args.window_snapshots),
            ndcg_k=int(args.ndcg_k),
            device=device,
            full_num_nodes=int(full_data.num_nodes),
            conversion_rate=int(conversion_rate),
            num_items=int(num_items),
            item_offset=int(item_offset),
            train_loader=train_loader,
            val_loader=val_loader,
            snapshots_loader=all_snapshots_loader,
            log_prefix=None,
        )

        bestm = out.get("best_val_metrics", {})
        print("\n===== FINAL RUN (best params) =====", flush=True)
        print(
            f"best_epoch={int(out['best_epoch'])} "
            f"val_NDCG={float(bestm.get('NDCG', out['best_val_ndcg'])):.6f} "
            f"val_Coverage={float(bestm.get('Coverage', 0.0)):.6f} "
            f"train_total_sec={float(out.get('train_total_sec', 0.0)):.2f} "
            f"avg_epoch_sec={float(out.get('avg_epoch_sec', 0.0)):.3f} "
            f"epochs_ran={int(float(out.get('epochs_ran', 0.0)))}",
            flush=True,
        )

        encoder = out["encoder"]
        decoder = out["decoder"]
        node_emb = out["node_emb"]

        test = eval_metrics(
            test_loader,
            all_snapshots_loader,
            encoder,
            decoder,
            node_emb,
            conversion_rate,
            num_items,
            item_offset,
            int(args.ndcg_k),
            int(args.window_snapshots),
        )

        print("\n===== TEST (after selecting best hyperparams) =====", flush=True)
        print(test, flush=True)
        return

    # single run mode (no optuna)
    print("===== SINGLE RUN (no Optuna) =====", flush=True)
    print(
        f"params: embedding_dim={emb_dim_fixed} num_hops={int(args.num_hops)} "
        f"dropout={float(args.dropout):.3f} lr={float(args.lr):.6g}",
        flush=True,
    )

    out = run_train_val(
        seed=int(args.seed),
        epochs=int(args.epochs),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        lr=float(args.lr),
        dropout=float(args.dropout),
        num_hops=int(args.num_hops),
        emb_dim=int(emb_dim_fixed),
        window_snapshots=int(args.window_snapshots),
        ndcg_k=int(args.ndcg_k),
        device=device,
        full_num_nodes=int(full_data.num_nodes),
        conversion_rate=int(conversion_rate),
        num_items=int(num_items),
        item_offset=int(item_offset),
        train_loader=train_loader,
        val_loader=val_loader,
        snapshots_loader=all_snapshots_loader,
        log_prefix="[single]",
    )

    bestm = out.get("best_val_metrics", {})
    print(
        f"[single] BEST: best_epoch={int(out['best_epoch'])} "
        f"best_val_NDCG={float(bestm.get('NDCG', out['best_val_ndcg'])):.6f} "
        f"best_val_Coverage={float(bestm.get('Coverage', 0.0)):.6f} "
        f"train_total_sec={float(out.get('train_total_sec', 0.0)):.2f} "
        f"avg_epoch_sec={float(out.get('avg_epoch_sec', 0.0)):.3f} "
        f"epochs_ran={int(float(out.get('epochs_ran', 0.0)))}",
        flush=True,
    )

    test = eval_metrics(
        test_loader,
        all_snapshots_loader,
        out["encoder"],
        out["decoder"],
        out["node_emb"],
        conversion_rate,
        num_items,
        item_offset,
        int(args.ndcg_k),
        int(args.window_snapshots),
    )

    print("\n===== TEST =====", flush=True)
    print(test, flush=True)


if __name__ == "__main__":
    main()