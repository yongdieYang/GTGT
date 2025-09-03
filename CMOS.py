import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear
from torch_geometric.nn.models.attentive_fp import GATEConv
from torch_geometric.nn import GATConv,  global_add_pool



class CrossModalAttention(torch.nn.Module):
    """Cross-attention module for cross-modal feature fusion."""
    def __init__(self, hidden_channels, num_heads=4):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        
        self.query_proj = torch.nn.Linear(hidden_channels, hidden_channels)
        self.key_proj = torch.nn.Linear(hidden_channels, hidden_channels)
        self.value_proj = torch.nn.Linear(hidden_channels, hidden_channels)
        
        self.out_proj = torch.nn.Linear(hidden_channels, hidden_channels)
        
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)
        
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels * 2, hidden_channels)
        )
        
    def forward(self, query, key, value):
        """
        Cross-attention forward pass.
        Args:
            query: [batch_size, hidden_channels] - Query features (e.g., molecular features)
            key: [batch_size, hidden_channels] - Key features (e.g., gene/taxonomy features)
            value: [batch_size, hidden_channels] - Value features (same as key)
        """
        batch_size = query.size(0)
        
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)
        
        Q = Q.view(batch_size, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)
        
        attention_scores = torch.sum(Q * K, dim=-1) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        attended_values = attention_weights.unsqueeze(-1) * V
        
        attended_output = attended_values.view(batch_size, self.hidden_channels)
        
        output = self.out_proj(attended_output)
        
        output = self.norm1(query + output)
        
        ffn_output = self.ffn(output)
        output = self.norm2(output + ffn_output)
        
        return output, attention_weights


class GATGENETAXONOMY_Enhanced(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
        taxonomy_dim: int = 750,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)

        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim,
                                  dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_conv.explain = False
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        # Gene processing layers with multi-convolution
        self.geneconv1 = torch.nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.geneconv2 = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.geneconv3 = torch.nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, stride=3)
        self.gene_adaptive_pool = torch.nn.AdaptiveAvgPool1d(hidden_channels)
        
        # Taxonomy data processing layer
        self.taxonomy_embedding = torch.nn.Linear(taxonomy_dim, hidden_channels)
        
        # Cross-attention modules
        self.gene_cross_attention = CrossModalAttention(hidden_channels)
        self.taxonomy_cross_attention = CrossModalAttention(hidden_channels)
        
        # Feature fusion layer
        self.feature_fusion = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 4, hidden_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels * 2, hidden_channels)
        )
        
        # Final linear layers
        self.lin4 = Linear(hidden_channels, hidden_channels // 2) 
        self.lin5 = Linear(hidden_channels // 2, out_channels)
        
        # For caching dynamically created layers
        self._duration_linear = None
        self._last_duration_dim = None
        
        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        # Reset multi-convolution layers
        self.geneconv1.reset_parameters()
        self.geneconv2.reset_parameters()
        self.geneconv3.reset_parameters()
        self.taxonomy_embedding.reset_parameters()
        self.lin4.reset_parameters()
        self.lin5.reset_parameters()
        if self._duration_linear is not None:
            self._duration_linear.reset_parameters()

    def _get_duration_linear(self, input_dim: int) -> torch.nn.Linear:
        """Dynamically create or get the duration linear layer."""
        if self._duration_linear is None or self._last_duration_dim != input_dim:
            self._duration_linear = torch.nn.Linear(input_dim, self.hidden_channels)
            self._last_duration_dim = input_dim
            # If on GPU, move to the corresponding device
            if next(self.parameters()).is_cuda:
                self._duration_linear = self._duration_linear.cuda()
        return self._duration_linear

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                batch: Tensor, gene: Tensor, taxonomy: Tensor, duration: Tensor) -> Tensor:
        
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        mol_features = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, mol_features), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            mol_features = self.mol_gru(h, mol_features).relu_()
        
        batch_size = mol_features.size(0)
        
        # Gene Embedding with multi-conv:
        if gene.dim() == 3:
            gene = F.relu(self.geneconv1(gene))
            gene = F.relu(self.geneconv2(gene))
            gene = self.geneconv3(gene).squeeze(1)
            gene_features = self.gene_adaptive_pool(gene.unsqueeze(1)).squeeze(1)
        elif gene.dim() == 2:
            gene_features = self.gene_adaptive_pool(gene.unsqueeze(1)).squeeze(1)
        else:
            gene = gene.view(batch_size, -1)
            if gene.size(1) != self.hidden_channels:
                if not hasattr(self, '_gene_linear') or self._gene_linear.in_features != gene.size(1):
                    self._gene_linear = torch.nn.Linear(gene.size(1), self.hidden_channels).to(gene.device)
                gene_features = self._gene_linear(gene)
            else:
                gene_features = gene
        
        # Taxonomy Embedding with proper batch handling:
        if taxonomy.dim() == 1:
            taxonomy = taxonomy.view(batch_size, -1)
        elif taxonomy.dim() == 2 and taxonomy.size(0) != batch_size:
            taxonomy = taxonomy.view(batch_size, -1)
        
        taxonomy_features = self.taxonomy_embedding(taxonomy)
        
        # Duration Embedding with proper batch handling:
        if duration.dim() == 1:
            duration = duration.view(batch_size, -1)
        elif duration.size(0) != batch_size:
            duration = duration.view(batch_size, -1)
            
        duration_dim = duration.size(-1)
        duration_linear = self._get_duration_linear(duration_dim)
        duration_features = duration_linear(duration).relu_()
        
        # Cross-attention driven feature fusion
        # Cross-attention between gene features and molecular features
        gene_attended, _ = self.gene_cross_attention(
            query=mol_features,
            key=gene_features,
            value=gene_features
        )
        
        # Cross-attention between taxonomy features and molecular features
        taxonomy_attended, _ = self.taxonomy_cross_attention(
            query=mol_features,
            key=taxonomy_features,
            value=taxonomy_features
        )
        
        # Feature fusion
        fused_features = torch.cat([
            mol_features,
            gene_attended,
            taxonomy_attended,
            duration_features
        ], dim=1)
        
        comprehensive_features = self.feature_fusion(fused_features)
        
        # Predictor:
        out = F.dropout(comprehensive_features, p=self.dropout, training=self.training)
        out = F.relu(self.lin4(out))
        out = F.dropout(out, p=self.dropout, training=self.training)
        final_output = self.lin5(out)
        
        return final_output

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'edge_dim={self.edge_dim}, '
                f'num_layers={self.num_layers}, '
                f'num_timesteps={self.num_timesteps}'
                f')')
