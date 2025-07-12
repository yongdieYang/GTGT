import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter
from torch_geometric.nn.models.attentive_fp import GATEConv
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool


class GATGENETAXONOMY(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
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
        self.mol_conv.explain = False  # Cannot explain global pooling.
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.geneconv = torch.nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, stride=3)
        self.gene_adaptive_pool = torch.nn.AdaptiveAvgPool1d(hidden_channels)
        
        self.taxonomy_adaptive_pool = torch.nn.AdaptiveAvgPool1d(hidden_channels)
        
        # Note: lin_duration is not predefined here, but is dynamically created in forward.
        
        self.lin4 = Linear(hidden_channels * 4, hidden_channels) 
        self.lin5 = Linear(hidden_channels, out_channels)
        
        self._duration_linear = None
        self._last_duration_dim = None
        
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.geneconv.reset_parameters()
        self.lin4.reset_parameters()
        self.lin5.reset_parameters()
        if self._duration_linear is not None:
            self._duration_linear.reset_parameters()

    def _get_duration_linear(self, input_dim: int) -> torch.nn.Linear:
        if self._duration_linear is None or self._last_duration_dim != input_dim:
            self._duration_linear = torch.nn.Linear(input_dim, self.hidden_channels)
            self._last_duration_dim = input_dim
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

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()
        
        batch_size = out.size(0)
        
        # Gene Embedding with proper batch handling:
        if gene.dim() == 3:  # [batch_size, channels, length]
            gene = self.geneconv(gene).squeeze(1)  # [batch_size, length]
            gene = self.gene_adaptive_pool(gene.unsqueeze(1)).squeeze(1)  # [batch_size, hidden_channels]
        elif gene.dim() == 2:  # [batch_size, length]
            gene = self.gene_adaptive_pool(gene.unsqueeze(1)).squeeze(1)  # [batch_size, hidden_channels]
        else:
            gene = gene.view(batch_size, -1)  # [batch_size, features]
            if gene.size(1) != self.hidden_channels:
                if not hasattr(self, '_gene_linear') or self._gene_linear.in_features != gene.size(1):
                    self._gene_linear = torch.nn.Linear(gene.size(1), self.hidden_channels).to(gene.device)
                gene = self._gene_linear(gene)
        
        # Taxonomy Embedding with proper batch handling:
        if taxonomy.dim() == 1:  
            taxonomy = taxonomy.view(batch_size, -1)  # [batch_size, features]
        elif taxonomy.dim() == 2 and taxonomy.size(0) != batch_size:
            taxonomy = taxonomy.view(batch_size, -1)

        if taxonomy.size(1) == self.hidden_channels:
            pass 
        elif taxonomy.dim() == 2:

            if not hasattr(self, '_taxonomy_linear') or self._taxonomy_linear.in_features != taxonomy.size(1):
                self._taxonomy_linear = torch.nn.Linear(taxonomy.size(1), self.hidden_channels).to(taxonomy.device)
            taxonomy = self._taxonomy_linear(taxonomy)
        else:

            taxonomy = self.taxonomy_adaptive_pool(taxonomy.unsqueeze(1)).squeeze(1)
        
        # Duration Embedding with proper batch handling:
        if duration.dim() == 1:
            duration = duration.view(batch_size, -1)
        elif duration.size(0) != batch_size:
            duration = duration.view(batch_size, -1)
            
        duration_dim = duration.size(-1)
        duration_linear = self._get_duration_linear(duration_dim)
        duration = duration_linear(duration).relu_()
        
        # Predictor:
        out = torch.cat((out, gene, taxonomy, duration), dim=1)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.lin4(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin5(out)


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'edge_dim={self.edge_dim}, '
                f'num_layers={self.num_layers}, '
                f'num_timesteps={self.num_timesteps}'
                f')')
