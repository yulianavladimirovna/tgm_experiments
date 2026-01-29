import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

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
     # return time bounds for event-ratio split
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    times = df_sorted[time_col].to_numpy() # timestamps of events
    n = len(times) # count events

    idx_val = int(n * train_ratio)
    idx_test = int(n * (train_ratio + val_ratio))

    val_time = times[idx_val]
    test_time = times[idx_test]

    return int(val_time), int(test_time)


def build_seen_local_by_user(df_edges: pd.DataFrame, num_users: int, item_offset: int) -> list:
    seen = [set() for _ in range(num_users)]
    for u, it in zip(df_edges["from"].to_numpy(), df_edges["to"].to_numpy()):
        if u < num_users and it >= item_offset:
            seen[u].add(int(it - item_offset))
    return [
        np.fromiter(s, dtype=np.int64) if len(s) else np.empty((0,), dtype=np.int64)
        for s in seen
    ]

parser = argparse.ArgumentParser(
    description="amazon beauty 5-core: GCN link prediction",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--seed", type=int, default=1337, help="random seed to use")
parser.add_argument("--device", type=str, default="cpu", help="torch device")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument("--n-layers", type=int, default=2, help="number of GCN layers")
parser.add_argument("--embed-dim", type=int, default=128, help="embedding dimension")
parser.add_argument(
    '--node-dim', type=int, default=256, help='node feat dimension if not provided'
)
parser.add_argument("--bsize", type=int, default=512, help="batch size")
parser.add_argument(
    "--path-dataset",
    type=str,
    default="data/amazon/beauty_5core_tgm.csv",
    help="CSV with columns: from,to,timestamp",
)
parser.add_argument(
    "--raw-time-gran",
    type=str,
    default="s",
    help="time unit of the raw timestamps in the CSV (passed to DGData.from_pandas)",
)
parser.add_argument(
    "--snapshot-time-gran",
    type=str,
    default="h"
)
parser.add_argument("--train-ratio", type=float, default=0.7)
parser.add_argument("--val-ratio", type=float, default=0.15)
parser.add_argument("--test-ratio", type=float, default=0.15)

parser.add_argument(
    "--ndcg-k",
    type=int,
    default=20,
    help="K for NDCG@K and Coverage@K",
)
parser.add_argument("--patience", type=int, default=30)
parser.add_argument("--min-delta", type=float, default=1e-4)
parser.add_argument(
    "--log-file-path", type=str, default=None, help="Optional path to write logs"
)

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
        self.in_channels = in_channels
        self.dropout = float(dropout)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, embed_dim))
        self.bns.append(torch.nn.BatchNorm1d(embed_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(embed_dim, embed_dim))
            self.bns.append(torch.nn.BatchNorm1d(embed_dim))
        self.convs.append(GCNConv(embed_dim, out_channels))

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, batch: DGBatch, node_feat: torch.Tensor) -> torch.Tensor:
        edge_index = torch.stack([batch.src, batch.dst], dim=0)
        # edge_index = torch.empty((2, 0), dtype=torch.long, device=batch.src.device)
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
    opt: torch.optim.Optimizer,
    conversion_rate: int,
) -> Tuple[float, torch.Tensor]:
    # updating z only when the current batch crosses the next snapshot boundary
    encoder.train()
    decoder.train()
    total_loss = 0.0
    total_count = 0
    static_node_feats = loader.dgraph.static_node_feats

    snapshots_iterator = iter(snapshots_loader)
    snapshot_batch = next(snapshots_iterator)

    z = encoder(snapshot_batch, static_node_feats)
    z = z.detach()

    for batch in tqdm(loader):
        opt.zero_grad()

        pos_out = decoder(z[batch.src], z[batch.dst])
        neg_out = decoder(z[batch.src], z[batch.neg])

        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))

        loss.backward()
        opt.step()

        bs = int(batch.src.shape[0])
        total_loss += float(loss.item()) * bs
        total_count += bs

        while batch.time[-1] > (snapshot_batch.time[-1] + 1) * conversion_rate:
            try:
                snapshot_batch = next(snapshots_iterator)
            except StopIteration:
                break
            z = encoder(snapshot_batch, static_node_feats)
            z = z.detach()

    epoch_loss = total_loss / max(total_count, 1)
    return epoch_loss, z



@log_gpu
@log_latency
@torch.no_grad()
def eval_metrics_full_ranking(
    loader,
    snapshots_loader,
    z: torch.Tensor,
    encoder,
    decoder,
    conversion_rate: int,
    num_items: int,
    item_offset: int,
    k: int,
    seen_local_by_user: Optional[list] = None,  # locals
) -> Dict[str, float]:

    encoder.eval()
    decoder.eval()

    static_node_feats = loader.dgraph.static_node_feats

    snapshots_iterator = iter(snapshots_loader)
    snapshot_batch = next(snapshots_iterator)

    mrr_list = []
    ndcg_list = []
    covered_items: Set[int] = set()

    device = loader.dgraph.device
    k_eff = min(int(k), int(num_items))

    # items global id
    item_ids_global = torch.arange(
        int(item_offset), int(item_offset + num_items),
        device=device, dtype=torch.long
    )

    for batch in tqdm(loader):
        # batch.src, batch.dst are positives (user->item edges)
        for i in range(int(batch.dst.numel())): # by count dsts
            u = int(batch.src[i].item())
            pos_gid = int(batch.dst[i].item())                    # global item id
            pos_local = int(pos_gid - int(item_offset))           # local in (0, num_items-1)

            u_z = z[u].view(1, -1)
            items_z = z[item_ids_global]

            # scores for all items
            logits = decoder(u_z.expand(int(num_items), -1), items_z).view(-1)

            # filtered full-ranking
            seen_local = seen_local_by_user[u]     # seen by user u (locals)
            if seen_local.size > 0:
                mask = torch.ones(int(num_items), device=device, dtype=torch.bool) # boolean, start with full true
                mask[torch.from_numpy(seen_local).to(device=device, dtype=torch.long)] = False # set false on seen items for user u
                mask[pos_local] = True                        # do not mask pos
                logits = logits.masked_fill(~mask, -float("inf")) # where false set inf to not recommend them

            # rank: how many items have score >= pos_score
            pos_score = logits[pos_local]
            rank = int((logits >= pos_score).sum().item())

            mrr_list.append(1.0 / float(rank))
            if rank <= k_eff:
                ndcg_list.append(1.0 / float(np.log2(rank + 1.0) + 1))
            else:
                ndcg_list.append(0.0)

            # top-K items for coverage
            topk_local = torch.topk(logits, k=k_eff, largest=True).indices  # (k_eff,)
            topk_global = (topk_local + int(item_offset)).detach().cpu().tolist()
            covered_items.update(topk_global)

        # update embeddings if we crossed snapshot boundary
        while batch.time[-1] > (snapshot_batch.time[-1] + 1) * conversion_rate:
            try:
                snapshot_batch = next(snapshots_iterator)
                z = encoder(snapshot_batch, static_node_feats).detach()
            except StopIteration:
                break

    mrr = float(np.mean(mrr_list)) if mrr_list else 0.0
    ndcg = float(np.mean(ndcg_list)) if ndcg_list else 0.0
    coverage = float(len(covered_items)) / float(max(1, num_items))
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

    def step(
        self,
        value: float,
        encoder: nn.Module,
        decoder: nn.Module,
        epoch: int,
    ) -> bool:
        if value > (self.best + self.min_delta):
            self.best = value
            self.best_epoch = int(epoch)
            self.cnt = 0

            self.best_encoder_state = {
                k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()
            }
            self.best_decoder_state = {
                k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()
            }
            return False

        self.cnt += 1
        return self.cnt >= self.patience # STOP if metric has not improved for patience epochs

    def restore_best(self, encoder: nn.Module, decoder: nn.Module) -> None:
        if self.best_encoder_state is None or self.best_decoder_state is None:
            return
        encoder.load_state_dict(self.best_encoder_state)
        decoder.load_state_dict(self.best_decoder_state)


# data loading
df = pd.read_csv(args.path_dataset)
df = df.dropna(subset=["from", "to", "timestamp"]).copy()
df["timestamp"] = df["timestamp"].astype("int64")

df, user_map, item_map = _build_bipartite_id_maps(df)
df = df.sort_values("timestamp").reset_index(drop=True)

# start at 0
t0 = int(df["timestamp"].min())
df["timestamp"] = (df["timestamp"] - t0).astype("int64")

num_users = len(user_map)
num_items = len(item_map)
item_offset = num_users

# only user->item for linkPredictor
full_data = DGData.from_pandas(
    edge_df=df,
    edge_src_col="from",
    edge_dst_col="to",
    edge_time_col="timestamp",
    edge_feats_col=None,
    time_delta=args.raw_time_gran,
)
if full_data.static_node_feats is None:
    full_data.static_node_feats = torch.randn(
        (full_data.num_nodes, args.node_dim), device=device
    )

# make bounds for splitting
val_time_q, test_time_q = bounds_event_ratio_split(
    df, args.train_ratio, args.val_ratio, time_col="timestamp"
)

train_data, val_data, test_data = full_data.split(
    TemporalSplit(
        val_time=int(val_time_q),
        test_time=int(test_time_q),
    )
)

train_dg = DGraph(train_data, device=device)
val_dg = DGraph(val_data, device=device)
test_dg = DGraph(test_data, device=device)

val_time = int(val_dg.start_time)
test_time = int(test_dg.start_time)

# seen on train
df_train_ui = df[df["timestamp"] < val_time].copy()
seen_train_local = build_seen_local_by_user(df_train_ui, num_users=num_users, item_offset=item_offset)

# seen on train+val
df_train_val_ui = df[df["timestamp"] < test_time].copy()
seen_train_val_local = build_seen_local_by_user(df_train_val_ui, num_users=num_users, item_offset=item_offset)


# make bidirectional edges ONLY for snapshots
df_rev = df.copy()
df_rev[["from", "to"]] = df_rev[["to", "from"]]
df_mp = pd.concat([df, df_rev], ignore_index=True).drop_duplicates(
    subset=["from", "to", "timestamp"]
)
df_mp = df_mp.sort_values("timestamp").reset_index(drop=True)

df_train_mp = df_mp[df_mp["timestamp"] < val_time].copy()
df_val_mp = df_mp[(df_mp["timestamp"] >= val_time) & (df_mp["timestamp"] < test_time)].copy()
df_test_mp = df_mp[df_mp["timestamp"] >= test_time].copy()

# DGData for snapshots (bidirectional)
train_mp = DGData.from_pandas(
    edge_df=df_train_mp,
    edge_src_col="from",
    edge_dst_col="to",
    edge_time_col="timestamp",
    edge_feats_col=None,
    time_delta=args.raw_time_gran,
)
val_mp = DGData.from_pandas(
    edge_df=df_val_mp,
    edge_src_col="from",
    edge_dst_col="to",
    edge_time_col="timestamp",
    edge_feats_col=None,
    time_delta=args.raw_time_gran,
)
test_mp = DGData.from_pandas(
    edge_df=df_test_mp,
    edge_src_col="from",
    edge_dst_col="to",
    edge_time_col="timestamp",
    edge_feats_col=None,
    time_delta=args.raw_time_gran,
)

# share the same node features everywhere
train_mp.static_node_feats = full_data.static_node_feats
val_mp.static_node_feats = full_data.static_node_feats
test_mp.static_node_feats = full_data.static_node_feats

# snapshots (discretization)
snapshot_td = TimeDeltaDG(args.snapshot_time_gran)
conversion_rate = int(snapshot_td.convert(train_dg.time_delta))

train_data_discretized = train_mp.discretize(args.snapshot_time_gran)
val_data_discretized = val_mp.discretize(args.snapshot_time_gran)
test_data_discretized  = test_mp.discretize(args.snapshot_time_gran)

train_snapshots = DGraph(train_data_discretized, device=device)
val_snapshots = DGraph(val_data_discretized, device=device)
test_snapshots  = DGraph(test_data_discretized, device=device)

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

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

train_snapshots_loader = DGDataLoader(
    train_snapshots, batch_unit=args.snapshot_time_gran
)
val_snapshots_loader = DGDataLoader(val_snapshots, batch_unit=args.snapshot_time_gran)
test_snapshots_loader = DGDataLoader(test_snapshots, batch_unit=args.snapshot_time_gran)



# model
encoder = GCNEncoder(
    in_channels=train_dg.static_node_feats_dim,
    embed_dim=args.embed_dim,
    out_channels=args.embed_dim,
    num_layers=args.n_layers,
    dropout=float(args.dropout),
).to(device)
decoder = LinkPredictor(node_dim=args.embed_dim, hidden_dim=args.embed_dim).to(device)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

stopper = EarlyStopper(patience=int(args.patience), min_delta=float(args.min_delta))

# train / val
z: Optional[torch.Tensor] = None

for epoch in range(1, int(args.epochs) + 1):
    with hm.activate(train_key):
        loss, z = train(
            train_loader,
            train_snapshots_loader,
            encoder,
            decoder,
            opt,
            conversion_rate,
        )

    assert z is not None

    with hm.activate(val_key):
        val = eval_metrics_full_ranking(
            val_loader,
            val_snapshots_loader,
            z,
            encoder,
            decoder,
            conversion_rate,
            num_items,
            item_offset,
            int(args.ndcg_k),
            seen_local_by_user=seen_train_local,
        )

    log_metric("Loss", float(loss), epoch=epoch)
    log_metric("Validation MRR", float(val["MRR"]), epoch=epoch)
    log_metric("Validation NDCG", float(val["NDCG"]), epoch=epoch)
    log_metric("Validation Coverage", float(val["Coverage"]), epoch=epoch)
    should_stop = stopper.step(float(val["NDCG"]), encoder, decoder, epoch)
    if should_stop:
        log_metric("EarlyStop", 1.0, epoch=epoch)
        break

# restore best
stopper.restore_best(encoder, decoder)
best_epoch = stopper.best_epoch if stopper.best_epoch >= 0 else epoch

with torch.no_grad():
    encoder.eval()
    static_node_feats = train_dg.static_node_feats

    last_snapshot_batch = None
    for last_snapshot_batch in train_snapshots_loader:
        pass
    z = encoder(last_snapshot_batch, static_node_feats).detach()

# test
with hm.activate(test_key):
    test = eval_metrics_full_ranking(
        test_loader,
        test_snapshots_loader,
        z,
        encoder,
        decoder,
        conversion_rate,
        num_items,
        item_offset,
        int(args.ndcg_k),
        seen_local_by_user=seen_train_val_local,
    )

log_metrics_dict({f"Test {k}": float(v) for k, v in test.items()}, epoch=best_epoch)