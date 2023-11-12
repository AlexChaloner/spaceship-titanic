import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data import CustomDataset, get_torch_data, get_train_data

# Combined model with embeddings and linear layer
class MixedModel(nn.Module):
    def __init__(self, num_categorical_cols, num_embeddings_list, embedding_dim_list, num_numeric_cols):
        super(MixedModel, self).__init__()

        # Embedding layers for categorical columns
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim)
            for num_embeddings, embedding_dim in zip(num_embeddings_list, embedding_dim_list)
        ])

        # Linear layer for numeric columns
        self.linear_layer = nn.Linear(num_numeric_cols, 1)

    def forward(self, x):
        # Separate categorical and numeric columns
        categorical_data = x[:, :len(self.embedding_layers)]
        numeric_data = x[:, len(self.embedding_layers):]

        # Apply embeddings to categorical data
        embedded_data = [embedding_layer(categorical_data[:, i]) for i, embedding_layer in enumerate(self.embedding_layers)]

        # Concatenate or combine the embeddings
        concatenated_embeddings = torch.cat(embedded_data, dim=1)

        # Combine embeddings with numeric data
        combined_data = torch.cat([concatenated_embeddings, numeric_data], dim=1)

        # Pass through the linear layer
        output = self.linear_layer(combined_data)

        return output
    

df = get_train_data()
# Hyperparameters
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
num_categorical_cols = len(categorical_cols)
num_embeddings_list = [len(df[col].unique()) for col in categorical_cols]
embedding_dim_list = [3, 2]
num_numeric_cols = len(df.columns) - num_categorical_cols


tensor_data = get_torch_data()
# Create a DataLoader
batch_size = 2
dataloader = DataLoader(tensor_data, batch_size=batch_size, shuffle=True)

# Initialize the model
model = MixedModel(num_categorical_cols, num_embeddings_list, embedding_dim_list, num_numeric_cols)

# Example usage in a loop
for batch in dataloader:
    # Forward pass through the model
    output = model(batch)
    # Your processing logic here
    print(output)
