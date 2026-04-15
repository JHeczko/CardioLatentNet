from torch import nn

from src.layers.mlp import FeedForwardLayer


class EncoderBlock(nn.Module):
    """A single Transformer encoder block.

        This block implements a standard Transformer encoder layer consisting of
        Multi-Head Self-Attention and a Feed-Forward MLP, integrated with
        Layer Normalization and dropout layers for regularization.

        Args:
            dim_hidden (int): The number of expected features in the input and output.
            num_heads (int): The number of heads in the multiheadattention model.
            dropout (float, optional): The dropout probability. Defaults to 0.2.

        Attributes:
            norm1 (nn.LayerNorm): Layer normalization applied before the attention mechanism.
            norm2 (nn.LayerNorm): Layer normalization applied before the MLP.
            attention (nn.MultiheadAttention): The multi-head attention module.
            mlp (FeedForwardLayer): The feed-forward network processing the attention output.
            att_dropout (nn.Dropout): Dropout layer applied to the attention output.
            mlp_dropout (nn.Dropout): Dropout layer applied to the MLP output.
    """

    def __init__(self, dim_hidden, num_heads, dropout=0.2):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim_hidden)
        self.norm2 = nn.LayerNorm(dim_hidden)

        self.attention = nn.MultiheadAttention(
            embed_dim=dim_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.mlp = FeedForwardLayer(
            dim_in=dim_hidden,
            dim_hidden=dim_hidden * 3
        )

        self.att_dropout = nn.Dropout(dropout)
        self.mlp_dropout = nn.Dropout(dropout)

    # x = (batch_size, seq_len, dim_hidden)
    # dim_hidden stays the same through whole encoder/decoder block
    def forward(self, x):
        """Processes the input through the self-attention and MLP blocks.

                Args:
                    x (torch.Tensor): The input tensor of shape (batch_size, seq_len, dim_hidden).

                Returns:
                    torch.Tensor: The output tensor of shape (batch_size, seq_len, dim_hidden),
                        preserving the original input dimensions via residual connections.
        """

        identity = x
        # x = (batch_size, seq_len, dim_embedded)
        x = self.norm1(x)
        # x = (batch_size, seq_len, dim_embedded)
        x, _ = self.attention(x, x, x, need_weights=False)
        # x = (batch_size, seq_len, dim_embedded)
        x = self.att_dropout(x)
        x = x + identity

        identity = x
        # x = (batch_size, seq_len, dim_embedded)
        x = self.norm2(x)
        # x = (batch_size, seq_len, dim_embedded)
        x = self.mlp(x)
        # x = (batch_size, seq_len, dim_embedded)
        x = self.mlp_dropout(x)
        x = x + identity

        return x