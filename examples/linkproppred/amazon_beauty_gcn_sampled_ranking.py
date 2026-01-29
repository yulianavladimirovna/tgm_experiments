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
from tgm.data import DGData, DGDataLoader, TemporalRatioSplit
from tgm.hooks import HookManager, NegativeEdgeSamplerHook
from tgm.nn import LinkPredictor
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric, log_metrics_dict
from tgm.util.seed import seed_everything


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
    default="h",
    help="time granularity to operate on for snapshots",
)
parser.add_argument("--train-ratio", type=float, default=0.7)
parser.add_argument("--val-ratio", type=float, default=0.15)
parser.add_argument("--test-ratio", type=float, default=0.15)
parser.add_argument(
    "--num-negs",
    type=int,
    default=50,
    help="number of negatives per positive for evaluation",
)
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


def _sample_negs_exclusive(
    pos_dst: torch.Tensor,
    num_items: int,
    item_offset: int,
    num_negs: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:

    pos_dst = pos_dst.to(torch.int64) # in global ids
    b = pos_dst.numel() # batch size (count of pos dst)

    # pos in local item space (0, num_items-1)
    pos_local = (pos_dst - int(item_offset)).view(-1, 1)

    # sample from (0, num_items-2)
    r = torch.randint(
        0,
        int(num_items) - 1,
        (int(b), int(num_negs)),
        device=pos_dst.device,
        dtype=torch.int64,
        generator=generator,
    )

    # map r to (0, num_items-1) skipping pos_local
    neg_local = r + (r >= pos_local).to(torch.int64)
    neg = neg_local + int(item_offset)
    return neg


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
    encoder.train()
    decoder.train()

    total_loss = 0.0
    total_count = 0  # per epoch

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
        loss = loss + F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))

        loss.backward()
        opt.step()

        batch_size = int(batch.src.shape[0])  # positive-edges count
        total_loss += float(loss) * batch_size
        total_count += batch_size

        # update embeddings if the prediction batch has moved to next snapshot
        while batch.time[-1] > (snapshot_batch.time[-1] + 1) * conversion_rate:
            try:
                snapshot_batch = next(snapshots_iterator)
                z = encoder(snapshot_batch, static_node_feats)
                z = z.detach()
            except StopIteration:
                pass

    epoch_loss = total_loss / total_count
    return epoch_loss, z


@log_gpu
@log_latency
@torch.no_grad()
def eval_metrics(
    loader: DGDataLoader,
    snapshots_loader: DGDataLoader,
    z: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    conversion_rate: int,
    num_items: int,
    item_offset: int,
    num_negs: int,
    k: int,
    seed_for_eval: int,
) -> Dict[str, float]:
    encoder.eval()
    decoder.eval()
    static_node_feats = loader.dgraph.static_node_feats

    snapshots_iterator = iter(snapshots_loader)
    snapshot_batch = next(snapshots_iterator)

    gen = torch.Generator(device=loader.dgraph.device)
    gen.manual_seed(int(seed_for_eval))

    mrr_list = []
    ndcg_list = []
    covered_items: Set[int] = set()

    k_eff = min(int(k), int(num_negs) + 1)

    for batch in tqdm(loader):
        neg_dst = _sample_negs_exclusive(
            batch.dst,
            num_items=num_items,
            item_offset=item_offset,
            num_negs=num_negs,
            generator=gen,
        )

        for idx, pos_dst in enumerate(batch.dst):
            query_src = batch.src[idx].repeat(int(num_negs) + 1)
            query_dst = torch.cat([pos_dst.view(1), neg_dst[idx]], dim=0)

            y_pred = decoder(z[query_src], z[query_dst]).sigmoid()  # (1 + num_negs,)

            # rank
            pos_score = y_pred[0]
            rank = 1 + int((y_pred[1:] >= pos_score).sum().item())
            # print(f'rank: {rank}')

            mrr_list.append(1.0 / float(rank))
            if rank <= k_eff:
                ndcg_list.append(1.0 / float(np.log2(rank + 1.0)))
            else:
                ndcg_list.append(0.0)

            topk_idx = torch.topk(y_pred, k=k_eff, largest=True).indices
            topk_items = query_dst[topk_idx]
            covered_items.update(topk_items.detach().cpu().tolist())

        # update embeddings if we crossed snapshot boundary
        while batch.time[-1] > (snapshot_batch.time[-1] + 1) * conversion_rate:
            try:
                snapshot_batch = next(snapshots_iterator)
                z = encoder(snapshot_batch, static_node_feats)
            except StopIteration:
                break

    mrr = float(np.mean(mrr_list)) if mrr_list else 0.0
    ndcg = float(np.mean(ndcg_list)) if ndcg_list else 0.0
    coverage = float(len(covered_items)) / float(max(1, num_items))

    return {"MRR": mrr, "NDCG": ndcg, "Coverage": coverage}


@dataclass
class EarlyStopper:
    patience: int
    min_delta: float
    best: float = float("-inf")
    bad_epochs: int = 0
    best_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

    def step(self, score: float, encoder: nn.Module, decoder: nn.Module) -> bool:
        improved = score > (self.best + self.min_delta)
        if improved:
            self.best = score
            self.bad_epochs = 0
            self.best_state = {
                "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                "decoder": {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()},
            }
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore_best(self, encoder: nn.Module, decoder: nn.Module, device: torch.device) -> None:
        if self.best_state is None:
            return
        encoder.load_state_dict({k: v.to(device) for k, v in self.best_state["encoder"].items()})
        decoder.load_state_dict({k: v.to(device) for k, v in self.best_state["decoder"].items()})

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

# only user->item for LinkPredictor
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

train_data, val_data, test_data = full_data.split(
    TemporalRatioSplit(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
)

train_dg = DGraph(train_data, device=device)
val_dg = DGraph(val_data, device=device)
test_dg = DGraph(test_data, device=device)

# make bidirectional edges only for snapshots
df_rev = df.copy()
df_rev[["from", "to"]] = df_rev[["to", "from"]]
df_mp = pd.concat([df, df_rev], ignore_index=True).drop_duplicates(
    subset=["from", "to", "timestamp"]
)
df_mp = df_mp.sort_values("timestamp").reset_index(drop=True)

# using ready-made split
val_time = int(val_dg.start_time)
test_time = int(test_dg.start_time)

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
test_data_discretized = test_mp.discretize(args.snapshot_time_gran)

train_snapshots = DGraph(train_data_discretized, device=device)
val_snapshots = DGraph(val_data_discretized, device=device)
test_snapshots = DGraph(test_data_discretized, device=device)

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

train_snapshots_loader = DGDataLoader(train_snapshots, batch_unit=args.snapshot_time_gran)
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
        val = eval_metrics(
            val_loader,
            val_snapshots_loader,
            z,
            encoder,
            decoder,
            conversion_rate,
            num_items,
            item_offset,
            int(args.num_negs),
            int(args.ndcg_k),
            seed_for_eval=int(args.seed),
        )

    log_metric("Loss", float(loss), epoch=epoch)
    log_metric("Validation MRR", float(val["MRR"]), epoch=epoch)
    log_metric("Validation NDCG", float(val["NDCG"]), epoch=epoch)
    log_metric("Validation Coverage", float(val["Coverage"]), epoch=epoch)

    if stopper.step(float(val["MRR"]), encoder, decoder):
        log_metric("EarlyStop", 1.0, epoch=epoch)
        break

# restore best
stopper.restore_best(encoder, decoder, device=device)

with torch.no_grad():
    encoder.eval()
    static_node_feats = train_dg.static_node_feats

    last_snapshot_batch = None
    for last_snapshot_batch in train_snapshots_loader:
        pass
    z = encoder(last_snapshot_batch, static_node_feats).detach()

# test
with hm.activate(test_key):
    test = eval_metrics(
        test_loader,
        test_snapshots_loader,
        z,
        encoder,
        decoder,
        conversion_rate,
        num_items,
        item_offset,
        int(args.num_negs),
        int(args.ndcg_k),
        seed_for_eval=int(args.seed) + 54321,
    )

log_metrics_dict({f"Test {k}": float(v) for k, v in test.items()}, epoch=epoch)
