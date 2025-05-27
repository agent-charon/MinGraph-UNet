import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple Graph Attention Layer, similar to original GAT paper.
    Assumes inputs are node features and an adjacency list (edge_index).
    """
    def __init__(self, in_features, out_features, dropout_rate, alpha, concat=True):
        """
        Args:
            in_features (int): Number of input features per node.
            out_features (int): Number of output features per node (for this head).
            dropout_rate (float): Dropout rate for attention coefficients and output features.
            alpha (float): Negative slope for LeakyReLU.
            concat (bool): If true, output is W*h'. If false (for last layer of multi-head),
                           output is averaged. Not used if this is a single head in a MultiHeadGATLayer.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.concat = concat # Note: concat is typically handled by the MultiHead wrapper

        # Learnable linear transformation
        self.W = nn.Linear(in_features, out_features, bias=False)
        # Learnable attention mechanism parameters a_1, a_2
        # Simpler: a single vector 'a' for [Wh_i || Wh_j]
        self.a = nn.Linear(2 * out_features, 1, bias=False)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(self.dropout_rate)

        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a.weight, gain=1.414)


    def forward(self, node_features, edge_index):
        """
        Args:
            node_features (torch.Tensor): Input node features (N, in_features), N = number of nodes.
            edge_index (torch.Tensor): Graph connectivity in COO format (2, E), E = number of edges.
                                       edge_index[0] = source nodes, edge_index[1] = target nodes.

        Returns:
            torch.Tensor: Output node features (N, out_features).
        """
        N = node_features.size(0) # Number of nodes

        # Apply linear transformation W to all node features
        h_transformed = self.W(node_features) # (N, out_features)
        
        # Prepare for attention score calculation
        # We need to compute e_ij = LeakyReLU(a^T [Wh_i || Wh_j]) for all connected i, j
        source_nodes_h = h_transformed[edge_index[0]] # (E, out_features) - features of source nodes for each edge
        target_nodes_h = h_transformed[edge_index[1]] # (E, out_features) - features of target nodes for each edge

        # Concatenate [Wh_i || Wh_j]
        edge_h_concat = torch.cat([source_nodes_h, target_nodes_h], dim=1) # (E, 2 * out_features)

        # Apply attention mechanism 'a'
        e = self.a(edge_h_concat) # (E, 1)
        e = self.leakyrelu(e)

        # Attention coefficients alpha_ij = softmax_j(e_ij)
        # Need to apply softmax grouped by target node j.
        # This requires careful indexing or a sparse softmax operation.
        # PyTorch Geometric handles this efficiently. Here's a manual way (can be slow for large graphs):
        
        attention_coeffs = torch.zeros(edge_index.shape[1], 1, device=node_features.device) # (E, 1)
        
        # A more efficient way to implement scatter_softmax if not using PyG:
        # 1. Subtract max for numerical stability: e_stable = e - scatter_max(e, edge_index[1], dim=0)[0][edge_index[1]]
        # 2. Exponentiate: exp_e = torch.exp(e_stable)
        # 3. Sum exponents per target node: exp_sum = scatter_add(exp_e, edge_index[1], dim=0, dim_size=N)[edge_index[1]]
        # 4. Normalize: attention_coeffs = exp_e / (exp_sum + 1e-8)
        
        # Using a simpler loop-based softmax for clarity here, but it's inefficient.
        # For production, use torch_geometric.utils.softmax
        # Or implement scatter_softmax as above.
        # Let's use a scatter_add approach for sum, then divide.
        
        # Numerator: exp(e_ij)
        exp_e = torch.exp(e - torch.max(e)) # Stability: subtract max(e) before exp
        
        # Denominator: sum_k exp(e_kj) for all k in N(j)
        # We need to sum exp_e for all incoming edges to node `j` (target nodes in edge_index[1])
        sum_exp_e_per_target_node = torch.zeros(N, 1, device=node_features.device)
        sum_exp_e_per_target_node.scatter_add_(0, edge_index[1].unsqueeze(1), exp_e) # Sum exp_e grouped by target node index

        # Get the correct sum for each edge's target node
        denominator = sum_exp_e_per_target_node[edge_index[1]]
        
        attention_coeffs = exp_e / (denominator + 1e-10) # (E, 1), adding epsilon for safety
        attention_coeffs = self.dropout(attention_coeffs) # Apply dropout to attention coefficients

        # Apply attention to source node features (Wh_i)
        # h'_j = sum_i alpha_ij * Wh_i
        # We need to aggregate weighted source_nodes_h for each target_node
        
        # Weighted features for each edge: attention_coeffs * source_nodes_h
        weighted_source_features = attention_coeffs * source_nodes_h # (E, out_features)

        # Aggregate these weighted features for each target node
        h_prime = torch.zeros_like(h_transformed) # (N, out_features)
        # Summing contributions for each target node j from its neighbors i
        # edge_index[1] gives target nodes, edge_index[0] gives source nodes.
        # We are calculating h_prime[j] = sum_{i in N(j)} alpha_ij * h_transformed[i]
        # So, we use edge_index[1] for indexing h_prime and aggregate weighted_source_features
        h_prime.scatter_add_(0, edge_index[1].unsqueeze(1).repeat(1, self.out_features), weighted_source_features)
        
        # if self.concat: # This is usually handled by a MultiHeadGAT layer
        #     return F.elu(h_prime) # Or a different activation like ReLU
        # else: # For averaging in the final layer of multi-head
        #     return h_prime
        return F.elu(h_prime) # Apply ELU activation (common in GAT)

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout_rate, alpha, concat=True):
        """
        Args:
            in_features (int): Input node feature dimensionality.
            out_features (int): Desired output node feature dimensionality (per head if concat=True,
                                or total if concat=False for final layer).
            num_heads (int): Number of attention heads.
            dropout_rate (float): Dropout for GAT layers.
            alpha (float): Negative slope for LeakyReLU in GAT.
            concat (bool): If True, outputs of heads are concatenated. If False (usually for final layer),
                           outputs are averaged.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        
        if concat:
            assert out_features % num_heads == 0, "out_features must be divisible by num_heads if concatenating"
            self.head_out_features = out_features // num_heads
        else:
            self.head_out_features = out_features # Each head outputs the full out_features

        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(
                GraphAttentionLayer(in_features, self.head_out_features, dropout_rate, alpha)
            )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, node_features, edge_index):
        head_outputs = [head(node_features, edge_index) for head in self.heads]
        
        if self.concat:
            # Concatenate features from all heads
            out = torch.cat(head_outputs, dim=1) # (N, num_heads * head_out_features)
        else:
            # Average features from all heads (typically for the final GAT layer)
            out = torch.mean(torch.stack(head_outputs, dim=0), dim=0) # (N, head_out_features)
        
        return self.dropout(out) # Apply dropout to the aggregated output

class GATNetwork(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, num_heads, num_gat_layers=1, dropout_rate=0.1, alpha=0.2):
        super().__init__()
        self.num_gat_layers = num_gat_layers
        self.gat_layers = nn.ModuleList()

        if num_gat_layers == 1:
            # Single GAT layer (possibly multi-head), outputting `output_dim`
            self.gat_layers.append(
                MultiHeadGATLayer(node_feature_dim, output_dim, num_heads, dropout_rate, alpha, concat=False) # Final layer averages
            )
        else:
            # First GAT layer
            self.gat_layers.append(
                MultiHeadGATLayer(node_feature_dim, hidden_dim, num_heads, dropout_rate, alpha, concat=True)
            )
            # Intermediate GAT layers (if any)
            for _ in range(num_gat_layers - 2):
                self.gat_layers.append(
                    MultiHeadGATLayer(hidden_dim * num_heads, hidden_dim, num_heads, dropout_rate, alpha, concat=True)
                )
            # Final GAT layer
            self.gat_layers.append(
                MultiHeadGATLayer(hidden_dim * num_heads, output_dim, num_heads, dropout_rate, alpha, concat=False) # Averages
            )
        
    def forward(self, node_features, edge_index):
        h = node_features
        for layer in self.gat_layers:
            h = layer(h, edge_index)
        return h


if __name__ == '__main__':
    # Example Usage
    N_nodes = 10
    in_dim = 32
    hidden_gat_dim = 64 # Per head if concatenating, or total if averaging
    out_dim_gat = 16 # Final output dimension from GAT block
    n_heads = 4
    dropout = 0.1
    leaky_alpha = 0.2

    # Dummy node features and edge index
    nodes = torch.randn(N_nodes, in_dim)
    # Create some random edges for a graph (ensure connected for meaningful test)
    # edge_idx = torch.randint(0, N_nodes, (2, N_nodes * 2)) # Random edges
    edge_idx = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 7, 8, 8, 9, 9, 4],
                             [1, 0, 2, 1, 3, 2, 0, 3, 5, 4, 6, 5, 8, 7, 9, 8, 4, 9]], dtype=torch.long)


    print("--- Testing Single GraphAttentionLayer ---")
    single_gat_head = GraphAttentionLayer(in_dim, hidden_gat_dim, dropout, leaky_alpha)
    out_single_head = single_gat_head(nodes, edge_idx)
    print(f"Single GAT head output shape: {out_single_head.shape}") # (N, hidden_gat_dim)

    print("\n--- Testing MultiHeadGATLayer (concat=True) ---")
    # If concat=True, out_features of MultiHeadGATLayer is num_heads * head_out_features
    # So, hidden_gat_dim here is the total output dim after concatenation
    multi_head_concat = MultiHeadGATLayer(in_dim, hidden_gat_dim, n_heads, dropout, leaky_alpha, concat=True)
    out_multi_concat = multi_head_concat(nodes, edge_idx)
    print(f"Multi-head GAT (concat) output shape: {out_multi_concat.shape}") # (N, hidden_gat_dim)
    assert out_multi_concat.shape[1] == hidden_gat_dim

    print("\n--- Testing MultiHeadGATLayer (concat=False, for final layer) ---")
    # If concat=False, out_features of MultiHeadGATLayer is the output dim per head, then averaged.
    # So, hidden_gat_dim here is the final desired output dim after averaging.
    multi_head_average = MultiHeadGATLayer(in_dim, out_dim_gat, n_heads, dropout, leaky_alpha, concat=False)
    out_multi_avg = multi_head_average(nodes, edge_idx)
    print(f"Multi-head GAT (average) output shape: {out_multi_avg.shape}") # (N, out_dim_gat)
    assert out_multi_avg.shape[1] == out_dim_gat

    print("\n--- Testing GATNetwork (num_gat_layers=1) ---")
    gat_net_1_layer = GATNetwork(in_dim, hidden_gat_dim, out_dim_gat, n_heads, num_gat_layers=1, dropout_rate=dropout, alpha=leaky_alpha)
    out_gat_net_1 = gat_net_1_layer(nodes, edge_idx)
    print(f"GATNetwork (1 layer) output shape: {out_gat_net_1.shape}") # (N, out_dim_gat)

    print("\n--- Testing GATNetwork (num_gat_layers=2) ---")
    # Layer 1: in_dim -> hidden_gat_dim (concatenated from heads)
    # Layer 2: hidden_gat_dim -> out_dim_gat (averaged from heads)
    # Note: GATNetwork internally handles head_out_features for MultiHeadGATLayer
    # If concat=True, hidden_dim in GATNetwork is total after concat.
    # If concat=False, output_dim in GATNetwork is total after averaging.
    gat_net_2_layers = GATNetwork(in_dim, hidden_dim=32, output_dim=out_dim_gat, num_heads=n_heads, num_gat_layers=2, dropout_rate=dropout, alpha=leaky_alpha)
    # For num_gat_layers=2:
    # Head 1: in_dim -> 32 (total, concat=True), so head_out = 32/num_heads
    # Head 2: 32 -> out_dim_gat (total, concat=False), so head_out = out_dim_gat
    out_gat_net_2 = gat_net_2_layers(nodes, edge_idx)
    print(f"GATNetwork (2 layers) output shape: {out_gat_net_2.shape}") # (N, out_dim_gat)