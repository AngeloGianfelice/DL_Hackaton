import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set, SAGPooling, ASAPooling, EdgePooling, TopKPooling
from source.conv import GNN_node, GNN_node_Virtualnode

class GNN(torch.nn.Module):
    def __init__(self, num_class=6, num_layer=5, emb_dim=300, 
                 gnn_type='gin', virtual_node=True, residual=False, 
                 drop_ratio=0.5, JK="last", graph_pooling="mean"):

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be > 1.")

        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual, gnn_type=gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual, gnn_type=gnn_type)

        # Setup pooling
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            gate_nn = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2 * emb_dim),
                torch.nn.BatchNorm1d(2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * emb_dim, 1)
            )
            self.pool = GlobalAttention(gate_nn=gate_nn)
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        elif self.graph_pooling == "sag":
            self.pool = SAGPooling(emb_dim, ratio=0.5)
        elif self.graph_pooling == "asap":
            self.pool = ASAPooling(emb_dim)
        elif self.graph_pooling == "edge":
            self.pool = EdgePooling(emb_dim)
        elif self.graph_pooling == "topk":
            self.pool = TopKPooling(emb_dim, ratio=0.5)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_class)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        if self.graph_pooling in ["sag", "asap"]:
            h_node, edge_index, _, batch, _, _ = self.pool(h_node, batched_data.edge_index, batch=batched_data.batch)
            h_graph = global_mean_pool(h_node, batch)

        elif self.graph_pooling == "edge":
            h_node, edge_index, batch, _ = self.pool(h_node, batched_data.edge_index, batched_data.batch)
            h_graph = global_mean_pool(h_node, batch)

        elif self.graph_pooling == "topk":
            h_node, edge_index, _, batch, perm, score = self.pool(h_node, batched_data.edge_index, batch=batched_data.batch)
            h_graph = global_mean_pool(h_node, batch)

        else:
            h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)