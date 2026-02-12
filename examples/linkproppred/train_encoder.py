import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple
from types import SimpleNamespace
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from tgm import DGBatch, DGraph, TimeDeltaDG
from tgm.data import DGData, DGDataLoader, TemporalSplit
from tgm.hooks import HookManager, NegativeEdgeSamplerHook
from tgm.nn import LinkPredictor
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric, log_metrics_dict
from tgm.util.seed import seed_everything


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


def _make_batch(edge_src: torch.Tensor, edge_dst: torch.Tensor, edge_time: torch.Tensor):
    return SimpleNamespace(edge_src=edge_src, edge_dst=edge_dst, edge_time=edge_time)


parser = argparse.ArgumentParser(
    description="Bipartite GCN link prediction (TGM)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--n-layers", type=int, default=2)
parser.add_argument("--embed-dim", type=int, default=128, help="GCN output dim (z dim)")
parser.add_argument("--node-dim", type=int, default=128, help="trainable node embedding dim (GCN input dim)")
parser.add_argument("--bsize", type=int, default=1, help="DGDataLoader batch_size (in batch_unit units)")
parser.add_argument("--path-dataset", type=str, default="data/amazon/beauty_5core_tgm.csv",
                    help="CSV with columns: from,to,timestamp")
parser.add_argument("--raw-time-gran", type=str, default="s",
                    help="raw timestamp unit for DGData.from_pandas")
parser.add_argument("--snapshot-time-gran", type=str, default="h",
                    help="time unit for batching events and snapshots")
parser.add_argument("--train-ratio", type=float, default=0.7)
parser.add_argument("--val-ratio", type=float, default=0.15)
parser.add_argument("--test-ratio", type=float, default=0.15)
parser.add_argument("--ndcg-k", type=int, default=20)
parser.add_argument("--patience", type=int, default=30)
parser.add_argument("--min-delta", type=float, default=1e-4)
parser.add_argument("--log-file-path", type=str, default=None)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)
seed_everything(args.seed)
device = torch.device(args.device)


class GCNEncoder(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            embed_dim: int,
            out_channels: int,
            num_layers: int,
            dropout: float,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, embed_dim))
        self.bns.append(torch.nn.BatchNorm1d(embed_dim))

        for _ in range(int(num_layers) - 2):
            self.convs.append(GCNConv(embed_dim, embed_dim))
            self.bns.append(torch.nn.BatchNorm1d(embed_dim))

        self.convs.append(GCNConv(embed_dim, out_channels))

    def forward(self, batch: DGBatch, node_feat: torch.Tensor) -> torch.Tensor:
        edge_index = torch.stack([batch.edge_src, batch.edge_dst], dim=0)
        x = node_feat
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


def _build_bipartite_id_maps(
        df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
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
    df["from"] = df["from"].astype(str).apply(map_user).astype("int64")

    u = len(user_map)
    df["to"] = df["to"].astype(str).apply(map_item).astype("int64")
    df["to"] = (df["to"] + u).astype("int64")
    return df, user_map, item_map


@log_gpu
@log_latency
def train(
        loader: DGDataLoader,
        snapshots_loader: DGDataLoader,
        encoder: nn.Module,
        decoder: nn.Module,
        node_emb: nn.Embedding,
        opt: torch.optim.Optimizer,
        conversion_rate: int,
) -> float:
    encoder.train()
    decoder.train()
    node_emb.train()

    total_loss = 0.0
    total_count = 0

    node_feat = node_emb.weight  # trainable

    # build cumulative snapshots
    snapshots_iterator = iter(snapshots_loader)
    try:
        snapshot_batch = next(snapshots_iterator)
    except StopIteration:
        raise RuntimeError("snapshots_loader is empty")

    cum_src = snapshot_batch.edge_src
    cum_dst = snapshot_batch.edge_dst
    cum_time = snapshot_batch.edge_time
    prev_snapshot_batch = _make_batch(cum_src, cum_dst, cum_time) # для энкодера берем все ребра до текущего момента

    for batch in tqdm(loader, desc="train", leave=False): # ребра внутри одного часа

        batch_start = int(batch.edge_time[0].item())
        # advance snapshots while the end of current snapshot bin <= batch_start
        while True:
            snap_idx = int(snapshot_batch.edge_time[-1].item()) # current snapshot
            snap_end = (snap_idx + 1) * int(conversion_rate) # in timestapms
            if snap_end <= batch_start: # текущий снапшот польностью позади батча
                prev_snapshot_batch = _make_batch(cum_src, cum_dst, cum_time)
                try:
                    snapshot_batch = next(snapshots_iterator) # берем новый снапшот
                except StopIteration:
                    break
                if snapshot_batch.edge_src.numel() > 0: # добавляем новый снапшот
                    cum_src = torch.cat([cum_src, snapshot_batch.edge_src], dim=0)
                    cum_dst = torch.cat([cum_dst, snapshot_batch.edge_dst], dim=0)
                    cum_time = torch.cat([cum_time, snapshot_batch.edge_time], dim=0)
            else:
                break

        opt.zero_grad(set_to_none=True)

        z = encoder(prev_snapshot_batch, node_feat)

        pos_out = decoder(z[batch.edge_src], z[batch.edge_dst])
        neg_out = decoder(z[batch.edge_src], z[batch.neg])

        # per-edge losses
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_out, torch.ones_like(pos_out), reduction="none"
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_out, torch.zeros_like(neg_out), reduction="none"
        )

        edge_loss = pos_loss + neg_loss

        # user-mean aggregation inside batch
        u = batch.edge_src
        uniq_u, inv = torch.unique(u, return_inverse=True)
        U = int(uniq_u.numel())

        sum_by_u = torch.zeros(U, device=edge_loss.device, dtype=edge_loss.dtype)
        cnt_by_u = torch.zeros(U, device=edge_loss.device, dtype=edge_loss.dtype)

        sum_by_u.scatter_add_(0, inv, edge_loss) # сумма лоссов
        cnt_by_u.scatter_add_(0, inv, torch.ones_like(edge_loss)) # кол-во ребер

        mean_by_u = sum_by_u / cnt_by_u.clamp_min(1.0)
        loss = mean_by_u.mean()

        loss.backward()
        opt.step()

        total_loss += float(loss.item()) * U
        total_count += U

    epoch_loss = total_loss / max(total_count, 1)
    return float(epoch_loss)


@log_gpu
@log_latency
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
) -> Dict[str, float]:
    encoder.eval()
    decoder.eval()
    node_emb.eval()

    node_feat = node_emb.weight

    # snapshots: build cumulative as in train
    snapshots_iterator = iter(snapshots_loader)
    try:
        snapshot_batch = next(snapshots_iterator)
    except StopIteration:
        raise RuntimeError("snapshots_loader is empty")

    cum_src = snapshot_batch.edge_src
    cum_dst = snapshot_batch.edge_dst
    cum_time = snapshot_batch.edge_time
    prev_snapshot_batch = _make_batch(cum_src, cum_dst, cum_time)

    prev_idx = int(prev_snapshot_batch.edge_time[-1].item()) # номер последнего snapshot, который сейчас включён в кумулятив
    z = encoder(prev_snapshot_batch, node_feat)

    device = loader.dgraph.device
    k_eff = min(int(k), int(num_items))

    # item nodes are [item_offset, item_offset + num_items)
    item_ids_global = torch.arange(
        int(item_offset), int(item_offset + num_items),
        device=device, dtype=torch.long
    )

    # per-user accumulators
    mrr_by_user = defaultdict(list)
    ndcg_by_user = defaultdict(list)

    # store ONE top-k list per user (last one)
    topk_by_user: Dict[int, torch.Tensor] = {}

    for batch in tqdm(loader, desc="eval", leave=False):
        batch_start = int(batch.edge_time[0].item())

        changed = False
        while True:
            snap_idx = int(snapshot_batch.edge_time[-1].item())
            snap_end = (snap_idx + 1) * int(conversion_rate)
            if snap_end <= batch_start:
                prev_snapshot_batch = _make_batch(cum_src, cum_dst, cum_time)
                changed = True

                try:
                    snapshot_batch = next(snapshots_iterator)
                except StopIteration:
                    break

                if snapshot_batch.edge_src.numel() > 0:
                    cum_src = torch.cat([cum_src, snapshot_batch.edge_src], dim=0)
                    cum_dst = torch.cat([cum_dst, snapshot_batch.edge_dst], dim=0)
                    cum_time = torch.cat([cum_time, snapshot_batch.edge_time], dim=0)
            else:
                break

        new_prev_idx = int(prev_snapshot_batch.edge_time[-1].item())
        if changed and new_prev_idx != prev_idx:
            z = encoder(prev_snapshot_batch, node_feat)
            prev_idx = new_prev_idx

        # aggregate per user
        n_edges = int(batch.edge_dst.numel())
        items_z = z[item_ids_global]  # эмбеддинги всех items
        for i in range(n_edges):
            u = int(batch.edge_src[i].item())
            pos_gid = int(batch.edge_dst[i].item())
            pos_local = int(pos_gid - int(item_offset))

            u_z = z[u].view(1, -1)
            logits = decoder(u_z.expand(int(num_items), -1), items_z).view(-1) # логит для юзера и всех товаров

            pos_score = logits[pos_local]
            rank = int((logits > pos_score).sum().item()) + 1

            mrr_by_user[u].append(1.0 / float(rank))
            ndcg_by_user[u].append(1.0 / float(np.log2(rank + 1.0)) if rank <= k_eff else 0.0)

            topk_local = torch.topk(logits, k=k_eff, largest=True).indices
            topk_by_user[u] = topk_local

    mrr_user_means = [float(np.mean(v)) for v in mrr_by_user.values() if len(v)]
    ndcg_user_means = [float(np.mean(v)) for v in ndcg_by_user.values() if len(v)]
    mrr = float(np.mean(mrr_user_means)) if mrr_user_means else 0.0
    ndcg = float(np.mean(ndcg_user_means)) if ndcg_user_means else 0.0

    # coverage union over users
    all_topk = torch.cat([t.view(-1) for t in topk_by_user.values()], dim=0)
    covered = int(torch.unique(all_topk).numel())
    coverage = float(covered) / float(max(1, int(num_items)))

    return {"MRR": mrr, "NDCG": ndcg, "Coverage": coverage}


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


df = pd.read_csv(args.path_dataset)
df = df.dropna(subset=["from", "to", "timestamp"]).copy()
df["timestamp"] = df["timestamp"].astype("int64")

df, user_map, item_map = _build_bipartite_id_maps(df)
df = df.sort_values("timestamp").reset_index(drop=True)

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

# event-ratio split -> time bounds
val_time_q, test_time_q = bounds_event_ratio_split(df, args.train_ratio, args.val_ratio, time_col="timestamp")

train_data, val_data, test_data = full_data.split(
    TemporalSplit(val_time=int(val_time_q), test_time=int(test_time_q))
)

train_dg = DGraph(train_data, device=device)
val_dg = DGraph(val_data, device=device)
test_dg = DGraph(test_data, device=device)

val_time = int(val_dg.start_time)
test_time = int(test_dg.start_time)

# build bidirectional edges for snapshots (message passing)
df_rev = df.copy()
df_rev[["from", "to"]] = df_rev[["to", "from"]]
df_mp = pd.concat([df, df_rev], ignore_index=True).drop_duplicates(subset=["from", "to", "timestamp"])
df_mp = df_mp.sort_values("timestamp").reset_index(drop=True)

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

# HookManager (NegativeEdgeSamplerHook)
hm = HookManager(keys=["train", "val", "test"])
hm.register(
    "train",
    NegativeEdgeSamplerHook(
        low=int(item_offset),
        high=int(item_offset + num_items),
        neg_ratio=1.0,
    ),
)
train_key, val_key, test_key = hm.keys

# events are batched by time unit (hours by default)
train_loader = DGDataLoader(train_dg, batch_size=int(args.bsize), batch_unit=args.snapshot_time_gran, hook_manager=hm)
val_loader = DGDataLoader(val_dg, batch_size=int(args.bsize), batch_unit=args.snapshot_time_gran, hook_manager=hm)
test_loader = DGDataLoader(test_dg, batch_size=int(args.bsize), batch_unit=args.snapshot_time_gran, hook_manager=hm)

node_emb = nn.Embedding(int(full_data.num_nodes), int(args.node_dim)).to(device)
torch.nn.init.normal_(node_emb.weight, std=0.1)

encoder = GCNEncoder(
    in_channels=int(args.node_dim),
    embed_dim=int(args.embed_dim),
    out_channels=int(args.embed_dim),
    num_layers=int(args.n_layers),
    dropout=float(args.dropout),
).to(device)
decoder = LinkPredictor(node_dim=int(args.embed_dim), hidden_dim=int(args.embed_dim)).to(device)
opt = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()) + list(node_emb.parameters()),
    lr=float(args.lr),
)

stopper = EarlyStopper(patience=int(args.patience), min_delta=float(args.min_delta))

for epoch in range(1, int(args.epochs) + 1):
    with hm.activate(train_key):
        loss = train(
            train_loader,
            all_snapshots_loader,
            encoder,
            decoder,
            node_emb,
            opt,
            conversion_rate,
        )

    with hm.activate(val_key):
        val = eval_metrics(
            val_loader,
            all_snapshots_loader,
            encoder,
            decoder,
            node_emb,
            conversion_rate,
            num_items,
            item_offset,
            int(args.ndcg_k),
        )

    log_metric("Loss", float(loss), epoch=epoch)
    log_metric("Validation MRR", float(val["MRR"]), epoch=epoch)
    log_metric("Validation NDCG", float(val["NDCG"]), epoch=epoch)
    log_metric("Validation Coverage", float(val["Coverage"]), epoch=epoch)

    should_stop = stopper.step(float(val["NDCG"]), encoder, decoder, node_emb, epoch)
    if should_stop:
        log_metric("EarlyStop", 1.0, epoch=epoch)
        break

# restore best weights
stopper.restore_best(encoder, decoder, node_emb)
best_epoch = stopper.best_epoch if stopper.best_epoch >= 0 else epoch

with hm.activate(test_key):
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
    )

log_metrics_dict({f"Test {k}": float(v) for k, v in test.items()}, epoch=best_epoch)