"""Directional distance encoding (DDE) of SubgraphRAG (Li et al., 2025), used by the SubgraphRAG and triple-MLP baselines."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing

class PEConv(MessagePassing):
    def __init__(self, eps=1e-12):
        super().__init__(aggr="add")
        self.eps = eps

    def forward(self, edge_index, x, edge_weight=None):
        num_edges = edge_index.size(1)

        if edge_weight is None:
            edge_weight = x.new_ones(num_edges)
        else:
            edge_weight = edge_weight.to(device=x.device, dtype=x.dtype)

        if edge_weight.dim() != 1 or edge_weight.numel() != num_edges:
            raise ValueError(
                "edge_weight must be a 1D tensor with one value per edge; "
                f"received shape {tuple(edge_weight.shape)} for {num_edges} edges"
            )

        weighted_sum = self.propagate(
            edge_index,
            x=x,
            edge_weight=edge_weight,
            size=(x.size(0), x.size(0)),
        )

        # edge_index[1] contains destination nodes under PyG's default
        # source-to-target message flow.
        destination = edge_index[1]
        weight_sum = x.new_zeros(x.size(0))
        weight_sum.index_add_(0, destination, edge_weight)

        # Nodes without incoming weight keep the zero output produced by the
        # additive aggregation. The mask also avoids division by zero.
        output = torch.zeros_like(weighted_sum)
        valid = weight_sum.abs() > self.eps
        output[valid] = weighted_sum[valid] / weight_sum[valid].unsqueeze(-1)
        return output

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.unsqueeze(-1)


class DDE(nn.Module):
    def __init__(self, num_rounds, num_reverse_rounds):
        super().__init__()

        self.layers = nn.ModuleList(PEConv() for _ in range(num_rounds))
        self.reverse_layers = nn.ModuleList(
            PEConv() for _ in range(num_reverse_rounds)
        )

    def forward(
        self,
        topic_entity_one_hot,
        edge_index,
        reverse_edge_index,
        edge_weight=None,
    ):
        if edge_weight is None:
            edge_weight = topic_entity_one_hot.new_ones(edge_index.size(1))

        result_list = []

        h_pe = topic_entity_one_hot
        for layer in self.layers:
            h_pe = layer(edge_index, h_pe, edge_weight=edge_weight)
            result_list.append(h_pe)

        h_pe_rev = topic_entity_one_hot
        for layer in self.reverse_layers:
            # reverse_edge_index is created by swapping the endpoints without
            # reordering edges, so the same edge_weight remains aligned.
            h_pe_rev = layer(
                reverse_edge_index,
                h_pe_rev,
                edge_weight=edge_weight,
            )
            result_list.append(h_pe_rev)

        return result_list
