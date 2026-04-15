from torch import nn


class LSTMConvBlock(nn.Module):
    def __init__(self,hidden_dim, ):
        super().__init__()

        self.