"""DLRM v2 handler — Recommendation Model (synthetic Criteo-like data)"""

import torch
import torch.nn as nn
from models.base import BaseModelHandler


class DLRMv2(nn.Module):
    """
    Simplified DLRM v2 architecture following the MLPerf reference.
    Supports dense + sparse (embedding) features with interaction layer.
    """

    def __init__(
        self,
        num_dense_features: int = 13,
        embedding_table_sizes: list = None,
        embedding_dim: int = 128,
        bottom_mlp_sizes: list = None,
        top_mlp_sizes: list = None,
    ):
        super().__init__()

        if embedding_table_sizes is None:
            # Criteo-like 26 sparse features, reduced sizes for demo
            embedding_table_sizes = [int(1e4)] * 26

        if bottom_mlp_sizes is None:
            bottom_mlp_sizes = [512, 256, embedding_dim]

        if top_mlp_sizes is None:
            # interaction output: embedding_dim + num_interactions
            num_sparse = len(embedding_table_sizes)
            num_interactions = (num_sparse + 1) * num_sparse // 2 + embedding_dim
            top_mlp_sizes = [512, 256, 1]

        # Bottom MLP
        layers = []
        in_size = num_dense_features
        for out_size in bottom_mlp_sizes:
            layers += [nn.Linear(in_size, out_size), nn.ReLU()]
            in_size = out_size
        self.bottom_mlp = nn.Sequential(*layers)

        # Embedding tables
        self.embeddings = nn.ModuleList([
            nn.EmbeddingBag(size, embedding_dim, mode="sum", sparse=True)
            for size in embedding_table_sizes
        ])

        # Interaction: concat + dot products
        self.embedding_dim = embedding_dim
        num_sparse = len(embedding_table_sizes)
        interact_out = (num_sparse + 1) * num_sparse // 2 + embedding_dim

        # Top MLP
        top_layers = []
        in_size = interact_out
        for out_size in top_mlp_sizes[:-1]:
            top_layers += [nn.Linear(in_size, out_size), nn.ReLU()]
            in_size = out_size
        top_layers.append(nn.Linear(in_size, top_mlp_sizes[-1]))
        self.top_mlp = nn.Sequential(*top_layers)

        self.sigmoid = nn.Sigmoid()

    def forward(self, dense_x, sparse_x, sparse_offsets):
        # Bottom MLP on dense features
        dense_out = self.bottom_mlp(dense_x)  # [B, emb_dim]

        # Embedding lookup
        emb_outs = []
        for i, emb in enumerate(self.embeddings):
            emb_outs.append(emb(sparse_x[:, i], sparse_offsets))  # [B, emb_dim]

        # Stack: [num_sparse+1, B, emb_dim]
        all_embs = torch.stack([dense_out] + emb_outs, dim=1)  # [B, num_sparse+1, emb_dim]

        # Dot product interactions
        B = all_embs.shape[0]
        interact = torch.bmm(all_embs, all_embs.transpose(1, 2))  # [B, N, N]
        N = all_embs.shape[1]
        # Upper triangle indices
        rows, cols = torch.triu_indices(N, N, offset=1)
        interact_flat = interact[:, rows, cols]  # [B, num_interactions]

        # Concat dense + interactions
        x = torch.cat([dense_out, interact_flat], dim=1)

        # Top MLP
        out = self.sigmoid(self.top_mlp(x))
        return out


class DLRMv2Handler(BaseModelHandler):
    NUM_DENSE = 13
    NUM_SPARSE = 26
    EMB_DIM = 128
    EMB_TABLE_SIZES = [int(1e4)] * 26

    def load(self):
        self.logger.info("  Building DLRM v2 (synthetic Criteo-like architecture)...")
        self.model = DLRMv2(
            num_dense_features=self.NUM_DENSE,
            embedding_table_sizes=self.EMB_TABLE_SIZES,
            embedding_dim=self.EMB_DIM,
        )
        self.model.eval()
        self.model.to(self.device)
        if self.dtype in (torch.float16, torch.bfloat16):
            # Keep embeddings in fp32, cast MLP layers only
            self.model.bottom_mlp = self.model.bottom_mlp.to(dtype=self.dtype)
            self.model.top_mlp = self.model.top_mlp.to(dtype=self.dtype)
            self.logger.info(f"  ℹ️  Embeddings kept in fp32, MLPs cast to {self.precision}")

        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"  ✅ DLRM v2 loaded | Params: {total_params/1e6:.1f}M")

    def prepare_data(self):
        B = self.batch_size
        # Dense features
        self._dense = torch.randn(B, self.NUM_DENSE, device=self.device, dtype=torch.float32)

        # Sparse features: each sample has 1 lookup per table → shape [B, num_sparse]
        self._sparse = torch.randint(
            0, min(self.EMB_TABLE_SIZES),
            (B, self.NUM_SPARSE),
            device=self.device,
        )

        # EmbeddingBag offsets: one lookup per row → sequential offsets
        self._offsets = torch.zeros(B, dtype=torch.long, device=self.device)

        self.logger.info(f"  ✅ Synthetic Criteo batch ready: dense={tuple(self._dense.shape)}, "
                         f"sparse={tuple(self._sparse.shape)}")

    def run_inference(self):
        with torch.no_grad():
            _ = self.model(self._dense, self._sparse, self._offsets)
        if torch.cuda.is_available() and str(self.device) != "cpu":
            torch.cuda.synchronize()