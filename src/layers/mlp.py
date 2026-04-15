from torch import nn
import torch


class FeedForwardLayer(nn.Module):
    """Standard Position-wise Feed-Forward Network for Transformer blocks.

        This module performs a two-stage transformation: it expands the input feature
        dimension to a higher-dimensional space using a linear projection, applies
        a GELU activation function, and then projects the features back to the
        original input dimension.

        Args:
            dim_in (int): The dimension of the input feature space.
            dim_hidden (int): The expanded dimension of the hidden layer.

        Attributes:
            l1 (nn.Linear): The first linear layer projecting from dim_in to dim_hidden.
            gelu (nn.GELU): The activation function applied to the hidden representation.
            l2 (nn.Linear): The second linear layer projecting back to dim_in.
        """
    def __init__(self, dim_in, dim_hidden):
        super().__init__()

        self.l1 = nn.Linear(dim_in, dim_hidden)

        self.gelu = nn.GELU()

        self.l2 = nn.Linear(dim_hidden, dim_in)
        self.l2.RESIDUAL_INIT = 1

    def forward(self, x):
        """Processes the input tensor through the MLP layers.

                Args:
                    x (torch.Tensor): The input tensor of shape (batch_size, seq_len, dim_in).

                Returns:
                    torch.Tensor: The output tensor of shape (batch_size, seq_len, dim_in).
        """

        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return x


if __name__ == '__main__':
    torch.manual_seed(0)

    batch_size = 1
    seq_len = 4
    dim = 4

    x = torch.randn(batch_size, seq_len, dim)

    print(x)

    mlp = FeedForwardLayer(dim, dim * 4)
    layer_norm = nn.LayerNorm(dim)

    x_out = layer_norm(mlp(x))

    print(x_out)
    print(x_out.mean(dim=-1))
    print(x_out.var(dim=-1, unbiased=False))