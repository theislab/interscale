# PyTorch

# PyTorch Lightning
from torch import nn
from torch_geometric.nn import GCNConv, MessagePassing

from interscale.module.base import LocalModule


class GCN(LocalModule):
    """Graph Convolutional Network (local module) - builds on PyTorch Geometric implementation."""

    def __init__(self, n_layers: int = 2, hidden_dim: int = 16, dropout_local: float = 0.1, **base_module_kwargs):

        super().__init__(**base_module_kwargs)

        self.module_name = "GCN"
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.norms = nn.ModuleList()
        self.dropout_local = dropout_local
        self.start_dropout = nn.Dropout(0)
        layers = []

        self.input_proj = nn.Linear(self.n_input, self.n_embed)
        self.input_norm = nn.LayerNorm(self.n_embed)

        in_dim = self.n_input
        hidden_dim = self.hidden_dim
        for _l_idx in range(n_layers - 1):
            layers += [
                GCNConv(in_channels=in_dim, out_channels=hidden_dim),
                nn.LayerNorm(hidden_dim),
                # nn.ReLU(inplace=True),
                nn.GELU(),
                nn.Dropout(self.dropout_local),
            ]

            in_dim = hidden_dim
        # layers += [GCNConv(in_channels=in_dim, out_channels=self.n_embed)]
        layers += [
            GCNConv(in_channels=in_dim, out_channels=self.n_embed),
            nn.LayerNorm(self.n_embed, elementwise_affine=False),
        ]
        self.layers = nn.ModuleList(layers)

        self.final_norm = nn.LayerNorm(self.n_embed, elementwise_affine=False)

    def forward(self, x, edge_index):
        """Compute node embeddings.

        Parameters
        ----------
        x
            Gene expression (``var × obs``).
        edge_index
            Adjacency matrix (``n × obs``).

        Returns
        -------
        Embeddings (``n × embed_dim``).
        """
        identity = self.input_proj(x)
        identity = self.input_norm(identity)
        for layer in self.layers:
            if isinstance(layer, MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)

        h = x + identity

        h = self.final_norm(h)

        # if self.training:
        #     print(f"SCALE CHECK:")
        #     print(f"  GCN Path      -> Mean: {x.mean().item():.4f}, Std: {x.std().item():.4f}")
        #     print(f"  Identity Path -> Mean: {identity.mean().item():.4f}, Std: {identity.std().item():.4f}")
        # h = F.gelu(x)
        return h

    def get_model_summary(self) -> str:
        """Return a string summary of the model's parameters."""
        summary = (
            f"GCN Local Component: \n"
            f"n_layers: {self.n_layers}, \n"
            f"n_hidden: {self.hidden_dim}, \n"
            f"n_embed: {self.n_embed}, \n"
            f"dropout_local: {self.dropout_local}"
        )
        return summary
