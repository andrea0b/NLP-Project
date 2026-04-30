# --- 5. SOTA IMPROVEMENT: Supervised Contrastive Learning (SCL) Dataset ---
try:
    import torch
    from torch.utils.data import Dataset
    
    class SCLFinancialDataset(Dataset):
        """
        PyTorch Dataset for Supervised Contrastive Learning (SCL).
        Automatically formats the processed dataframe for SupConLoss training.
        Requires ternary labels to be shifted from (-1, 0, 1) -> (0, 1, 2)
        """
        def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
            self.texts = df['bert_input'].tolist()
            # Map ternary labels: -1 -> 0, 0 -> 1, 1 -> 2
            self.labels = (df['label'] + 1).astype(int).tolist() 
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            item = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            # Remove batch dimension added by the tokenizer's default behavior
            item = {key: val.squeeze(0) for key, val in item.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item
except ImportError:
    pass