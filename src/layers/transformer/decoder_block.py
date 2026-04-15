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
            gating (bool): Determine if use gating systems from U-skip connections.

        Attributes:
            norm1, norm2, norm3 (nn.LayerNorm): Normalization layers for sub-layers.
            dropout1, dropout2, dropout3 (nn.Dropout): Regularization layers.
            self_att (nn.MultiheadAttention): Self-attention mechanism for the decoder.
            cross_att (nn.MultiheadAttention): Cross-attention for encoder-decoder fusion.
            mlp (FeedForwardLayer): Feed-forward network for feature refinement.
    """
    def __init__(self, dim_hidden, num_heads, dropout=0.2, gating=False):
        super().__init__()

        self.gating = gating

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

        # gating mechanizm
        if gating:
            self.gate = nn.Linear(dim_hidden, 1)
            self.gate.SKIP_INIT = 1
            nn.init.constant_(self.gate.bias, -1)

    # x = (batch_size, seq_len, dim_hidden)
    # enc_out = (batch_size, compressed_seq_len, dim_hidden)
    def forward(self, x, enc_out):
        """Processes input features through self-attention, cross-attention, and MLP. Using learned gating to say how much encoder input to take

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
        if self.gating:
            g = torch.sigmoid(self.gate(x.mean(dim=1, keepdim=True)))
        else:
            g = 1.0
        x = x * g
        x = self.dropout2(x)
        x = x + identity

        identity = x
        x = self.norm3(x)
        x = self.mlp(x)
        x = self.dropout3(x)
        x = x + identity

        return x


if __name__ == '__main__':
    __w_g = nn.Linear(128, 1)
    gate = lambda x: torch.sigmoid(__w_g(torch.mean(x, dim=1)))

    t = torch.randn(4, 60, 128)
    t = torch.mean(t, dim=1)
    t_1 = __w_g(t)
    t_2 = torch.sigmoid(t_1)
    print(t_2)