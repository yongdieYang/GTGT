import os
from typing import Optional, Callable
import torch
from torch_geometric.data import InMemoryDataset


class ToxicDataset(InMemoryDataset):
    """Dataset class for handling toxicity data, supporting multiple feature processing modes."""

    GENE_MAP = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    DURATION_MAP = [24, 48, 72, 96]  # Adjust these values based on your actual data.

    def __init__(
            self,
            root: str,
            filenames: str,
            mode: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
    ):
        """
        Initializes the toxicity dataset.

        Args:
            root: Root directory for the data.
            filenames: Raw data filenames.
            mode: Processing mode, can be 'gene', 'taxonomy', 'gene+taxonomy', or 'noclass'.
            transform: Data transformation function.
            pre_transform: Data pre-transformation function.
            pre_filter: Data pre-filtering function.
        """
        self.filenames = filenames
        self.mode = mode
        self.validate_mode()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def validate_mode(self):
        """Validates the mode parameter."""
        valid_modes = ['gene', 'taxonomy', 'gene+taxonomy', 'noclass']
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Mode must be one of: {valid_modes}")

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, f'{self.mode}_processed')

    @property
    def raw_file_names(self) -> str:
        """The name of the raw files. If found, download is skipped."""
        return self.filenames

    @property
    def processed_file_names(self) -> str:
        """The name of the processed files. If found in processed_dir, processing is skipped."""
        return 'data.pt'

    def process(self):
        import pandas as pd
        from torch_geometric.utils import from_smiles
        
        print("Processing dataset in {} mode...".format(self.mode))
        data_list = []
        
        df = pd.read_csv(self.raw_paths[0])
        print(f"CSV file has {len(df.columns)} columns and {len(df)} rows")
        
        for idx, row in df.iterrows():
            try:
                organism = row.iloc[0]
                chemical_name = row.iloc[1] 
                duration = int(row.iloc[2])
                smiles = row.iloc[3]
                mg_perl = float(row.iloc[4])
                gene_data = row.iloc[5]
                
                available_cols = len(row)
                if available_cols < 756:
                    print(f"Warning: Expected 756 columns but found {available_cols}")
                    taxonomy_data = row.iloc[6:].values.astype(float)
                else:
                    taxonomy_data = row.iloc[6:756].values.astype(float)
                
                data = from_smiles(smiles)
                if data is None:
                    print(f"Warning: Could not generate graph from SMILES for row {idx}")
                    continue
                
                data.organism = organism
                data.chemical_name = chemical_name
                data.smiles = smiles
                
                duration_tensor = torch.zeros(len(self.DURATION_MAP))
                if duration in self.DURATION_MAP:
                    duration_tensor[self.DURATION_MAP.index(duration)] = 1.0
                data.duration = duration_tensor
                
                data.y = torch.tensor([mg_perl], dtype=torch.float)

                if self.mode in ['gene', 'gene+taxonomy']:
                    data.gene = self._process_gene_data(gene_data)

                if self.mode in ['taxonomy', 'gene+taxonomy']:
                    data.taxonomy = torch.tensor(taxonomy_data, dtype=torch.float)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

        print(f"Successfully processed {len(data_list)} samples")
        
        if len(data_list) == 0:
            raise ValueError("No valid samples were processed. Please check your CSV file format.")
        
        torch.save(self.collate(data_list), self.processed_paths[0])

    def _process_gene_data(self, gene_data: str) -> torch.Tensor:
        """Processes gene data and converts it to a tensor."""
        encoded_gene_data = [self.GENE_MAP[char] for char in gene_data]
        return torch.tensor(encoded_gene_data, dtype=torch.float).transpose(0, 1).unsqueeze(0)
