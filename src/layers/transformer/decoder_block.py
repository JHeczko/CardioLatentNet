import torch
from torch import nn

from src.layers import FeedForwardLayer

class DecoderBlock(nn.Module):
    """A single decoder block for a U-shaped Transformer architecture.

        This block integrates information from the current decoder state and
        the encoder skip-connection through sequential attention mechanisms.

        Args:
            dim_hidden (int): Dimensionality of the hidden states (embedding size).
            num_heads (int): Number of attention heads for self and cross-attention.
            dropout (float): Dropout probability applied after each sub-layer.

        Attributes:
            norm1, norm2, norm3 (nn.LayerNorm): Normalization layers for sub-layers.
            dropout1, dropout2, dropout3 (nn.Dropout): Regularization layers.
            self_att (nn.MultiheadAttention): Self-attention mechanism for the decoder.
            cross_att (nn.MultiheadAttention): Cross-attention for encoder-decoder fusion.
            mlp (FeedForwardLayer): Feed-forward network for feature refinement.
    """
    def __init__(self, dim_hidden, num_heads, dropout=0.2):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim_hidden)
        self.norm2 = nn.LayerNorm(dim_hidden)
        self.norm3 = nn.LayerNorm(dim_hidden)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.self_att = nn.MultiheadAttention(
            embed_dim=dim_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True)

        self.cross_att = nn.MultiheadAttention(
            embed_dim=dim_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.mlp = FeedForwardLayer(dim_in=dim_hidden, dim_hidden=dim_hidden * 3)

    # x = (batch_size, seq_len, dim_hidden)
    # enc_out = (batch_size, compressed_seq_len, dim_hidden)
    def forward(self, x, enc_out):
        """Processes input features through self-attention, cross-attention, and MLP.

                Args:
                    x (torch.Tensor): Decoder input tensor (batch_size, seq_len, dim_hidden).
                    enc_out (torch.Tensor): Encoder output for skip-connection
                        (batch_size, compressed_seq_len, dim_hidden).

                Returns:
                    torch.Tensor: Refined decoder features of shape (batch_size, seq_len, dim_hidden).
        """
        identity = x
        x = self.norm1(x)
        x, _ = self.self_att(x, x, x, need_weights=False)
        x = self.dropout1(x)
        x = x + identity

        identity = x
        x = self.norm2(x)
        x, _ = self.cross_att(x, enc_out, enc_out, need_weights=False)
        x = self.dropout2(x)
        x = x + identity

        identity = x
        x = self.norm3(x)
        x = self.mlp(x)
        x = self.dropout3(x)
        x = x + identity

        return x
