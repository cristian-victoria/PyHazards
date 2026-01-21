import torch
import torch.nn as nn 
import torch.nn.functional as F
import math

#inital code

class HydroGraphNet(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        harmonics: int = 5,
        num_gn_blocks: int = 5,
    ):
        super().__init__()
        self.edge_in_dim = edge_in_dim

        #-------encoder-------
        self.node_encoder = KAN(in_dim = node_in_dim, hidden_dim = hidden_dim, harmonics = harmonics)
        self.edge_encoder = MLP(in_dim= edge_in_dim, out_dim = hidden_dim , hidden_dim= hidden_dim)
        #------processor-------
        self.processor = nn.ModuleList(
            [
                GN(hidden_dim = hidden_dim)
                for i in range(num_gn_blocks)
            ]
        )
        #-------decoder-------
        self.decoder = MLP(in_dim = hidden_dim, out_dim = out_dim)
    def forward(self, batch):
        # batch is a dict from graph_collate
        x = batch["x"]        # (B, past_days, N, F)
        adj = batch["adj"]    # (B, N, N) or None

        # use last timestep
        node_x = x[:, -1]     # (B, N, F)

        # ---- build senders / receivers from adjacency ----
        if adj is None:
            raise ValueError("HydroGraphNet requires adjacency")

        # assume same adjacency for whole batch
        A = adj[0]            # (N, N)
        senders, receivers = A.nonzero(as_tuple=True)

        # ---- encoder ----
        node = self.node_encoder(node_x)

        # simple edge features (ones) for now
        edge_x = torch.ones(
            node.size(0),
            senders.numel(),
            self.edge_in_dim,
            device=node.device
        )

        edge = self.edge_encoder(edge_x)

        # ---- GN processor ----
        for gn in self.processor:
            node, edge = gn(node, edge, senders, receivers)

        # ---- decoder ----
        out = self.decoder(node)
        return out


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64): 
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class KAN(nn.Module):
    def __init__(
        self,
        in_dim: int, 
        harmonics: int = 5,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.harmonics = harmonics

        self.feature_proj = nn.ModuleList(
            [
                nn.Linear(2*harmonics + 1, hidden_dim)
                for i in range(in_dim)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []

        for i in range(self.in_dim):
            xi = x[:, : ,i].unsqueeze(-1)

           
            basis = [torch.ones_like(xi)]
            for k in range(1, self.harmonics + 1):
                basis.append(torch.sin(k * xi))
                basis.append(torch.cos(k * xi))

            basis = torch.cat(basis, dim=-1)

          
            outputs.append(self.feature_proj[i](basis))
      
        return torch.stack(outputs, dim=0).sum(dim=0)
 
class GN(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_mlp = MLP(3 * hidden_dim, hidden_dim, hidden_dim)
        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim)

    def forward(self, node, edge, senders, receivers):
        sender_feat = node[:, senders, :]
        receiver_feat = node[:, receivers, :]

        edge_input = torch.cat([edge, sender_feat, receiver_feat], dim=-1)
        edge = edge + self.edge_mlp(edge_input)

        agg = torch.zeros_like(node)
        agg.index_add_(1, receivers, edge)

        node_input = torch.cat([node, agg], dim=-1)
        node = node + self.node_mlp(node_input)

        return node, edge


class HydroGraphLoss(nn.Module):
    "PREDICTION"
    def __init__(self):
        super().__init__()
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


def hydrographnet_builder(
    task: str,
    node_in_dim: int,
    edge_in_dim: int,
    out_dim: int,
    **kwargs,
):
    if task != "regression":
        raise ValueError("HydroGraphNet only supports regression")

    return HydroGraphNet(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        out_dim=out_dim,
        hidden_dim=kwargs.get("hidden_dim", 64),
        harmonics=kwargs.get("harmonics", 5),
        num_gn_blocks=kwargs.get("num_gn_blocks", 5),
    )
