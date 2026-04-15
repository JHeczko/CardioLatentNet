from torch import nn
import torch

class PositionalEncoding(nn.Module):
    """Learned positional encoding layer.

        This module assigns a unique, learnable embedding vector to each position
        in a sequence, allowing the model to incorporate order information.

        Args:
            max_context_length (int): Maximum sequence length supported.
            dim_embedded (int): Dimensionality of the embedding vectors.

        Attributes:
            position_embedding (nn.Embedding): Lookup table for position embeddings.
    """
    def __init__(self, max_context_length, dim_embedded):
        super().__init__()
        self.position_embedding = nn.Embedding(max_context_length, dim_embedded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional information to input tensors.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_hidden).

                Returns:
                    torch.Tensor: Position embeddings of shape (batch_size, seq_len, dim_embedded).
        """

        batch_size, seq_len, _ = x.shape
        # pos = (, context_len)
        pos = torch.arange(seq_len, device=x.device, dtype=torch.long)

        # pos_emb = (, context_len, embedded_dim)
        pos_emb = self.position_embedding(pos)
        pos_emb = pos_emb.repeat(batch_size, 1).view(batch_size, seq_len, -1)

        return pos_emb




if __name__ == '__main__':
    embedding = PositionalEncoding(60, 128)

    dummy_vector = torch.ones(30, 60, 128, dtype=torch.int)

    print(dummy_vector.shape)

    out_embedding = embedding(dummy_vector)
    print(out_embedding)
    # torch.Size([4, 51212, 512])
    print(out_embedding.shape)