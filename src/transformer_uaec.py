import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .utils.config.model import TransformerAecConfig
from .layers.blocks import TransformerEncoderBlock, TransformerDecoderBlock
from .layers.encoding import PositionalEncoding
from .layers.dimension import Upsampler, Downsampler
from .layers.attention_pooling import AttentionPooling

class TransformerAec(nn.Module):
    """Transformer-based U-shaped Autoencoder (UAEC).

        This architecture employs a symmetric encoder-decoder structure with
        skip-connections (U-net style) to process sequential data, utilizing
        downsampling and upsampling layers to manage latent representations.

        Args:
            blocks (int): Number of main blocks in the architecture.
            enc_dec_ratio (tuple): Ratio of encoders to decoders per block.
            num_att_heads (int): Number of heads for multi-head attention.
            input_dim (int): Dimensionality of the raw input.
            hidden_dim (int): Internal dimensionality of the Transformer layers.
            seq_len (int): Length of the input sequence.
    """

    def __init__(self, config: TransformerAecConfig):
        super().__init__()

        blocks = config.blocks
        enc_dec_ratio = config.enc_dec_ratio
        num_att_heads = config.num_att_heads
        input_dim = config.input_dim
        hidden_dim = config.hidden_dim
        latent_dim = config.latent_dim
        seq_len = config.seq_len
        dropout = config.dropout
        gradient_checkpointing = config.gradient_checkpointing


        self.grad_checkpointing = gradient_checkpointing

        self.encoders_per_block = enc_dec_ratio[0]
        self.decoders_per_block = enc_dec_ratio[1]
        num_encoders = blocks*enc_dec_ratio[0]
        num_decoders = blocks*enc_dec_ratio[1]

        # ============== ENCODER PART ==============
        self.encoder_embedding = nn.Linear(input_dim, hidden_dim)
        self.encoder_pos_enc = PositionalEncoding(max_context_length=seq_len, dim_embedded=hidden_dim)

        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        for i in range(num_encoders):
            self.encoders.append(TransformerEncoderBlock(dim_hidden=hidden_dim, num_heads=num_att_heads, dropout=dropout))
            if((i+1)%self.encoders_per_block == 0):
                self.downsamplers.append(Downsampler(hidden_dim=hidden_dim, stride=2,kernel_size=3))

        # ============== LATENT PART ==============
        # encoder strona
        self.latent_pool = AttentionPooling(hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)

        # decoder strona — odwrotność
        self.latent_unproj = nn.Linear(latent_dim, hidden_dim)
        # seq_len po wszystkich downsamplach to właśnie te 5
        self.bottleneck_seq_len = seq_len // (2 ** blocks)

        # ============== DECODER PART ==============
        self.dec_pos_emb = PositionalEncoding(max_context_length=seq_len, dim_embedded=hidden_dim)

        self.connect_decoder = TransformerDecoderBlock(dim_hidden=hidden_dim, num_heads=num_att_heads, dropout=dropout)

        self.decoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for i in range(num_decoders):
            self.decoders.append(TransformerDecoderBlock(dim_hidden=hidden_dim, num_heads=num_att_heads, dropout=dropout))
            if ((i + 1) % self.decoders_per_block == 0):
                self.upsamplers.append(Upsampler(hidden_dim=hidden_dim, stride=2,kernel_size=3))

        # ============== FINAL PROJECTION ==============
        self.proj = nn.Linear(hidden_dim, input_dim)

        # ============== WEIGHT INIT ==============
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():

            # --- GATE ---
            if isinstance(m, nn.Linear) and hasattr(m, 'GATE'):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, -1.0)

            # --- LINEAR / EMBEDDING ---
            elif isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

            # --- CONV ---
            elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            # --- LAYERNORM ---
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # INPUT - x = (batch_size, seq_len, channels)
    def forward(self, x):
        """Performs the forward pass of the U-shaped Autoencoder.

                Args:
                    x (torch.Tensor): Input tensor (batch_size, seq_len, input_dim).

                Returns:
                    torch.Tensor: Reconstructed output tensor (batch_size, seq_len, input_dim).
        """

        # ----- ENCODER PASS -----
        # x = (batch_size, seq_len, hidden_dim)
        enc_out = self.encoder_embedding(x)
        starting_shape = enc_out.shape
        # pos_enc = (batch_size, seq_len, hidden_dim)
        enc_pos_enc = self.encoder_pos_enc(x)
        # adding learned position encoding
        enc_out = enc_out + enc_pos_enc


        encs = []
        # x = (batch_size, seq_len/(num_encoder//2), hidden_dim)
        downsamplers_iter = iter(self.downsamplers)
        for i,encoder_block in enumerate(self.encoders):
            if self.grad_checkpointing and self.training:
                enc_out = torch.utils.checkpoint.checkpoint(encoder_block, enc_out, use_reentrant=False)
            else:
                enc_out = encoder_block(enc_out)

            if ((i + 1) % self.encoders_per_block == 0):
                # doing append before downsampling
                encs.append(enc_out)

                # downsampling
                downsampler = next(downsamplers_iter)
                enc_out = downsampler(enc_out)

        # we have to reverse, cuz it was being added in reversed order
        # now we have nicely saved encoders output for U-shape AEC
        encs = list(reversed(encs))

        # ----- LATENT PASS -----
        # after all encoder blocks
        # enc_out: (B, 5, 128)
        latent = self.latent_pool(enc_out)  # (B, 128)
        latent = self.latent_proj(latent)  # (B, latent_dim)

        # clear latent vector with Attention Pooling

        # then we have to back the size in order to make decoder pass
        bottleneck_seq_len = enc_out.shape[1]
        x = self.latent_unproj(latent)  # (B, 128)
        x = x.unsqueeze(1).expand(-1, bottleneck_seq_len, -1)  # (B, 5, 128)

        # ----- DECODER PASS -----
        # dec_out = (batch_size, seq_len/(num_encoder//2), hidden_dim)

        # adding postion emb
        dec_out = x
        pos = self.dec_pos_emb(dec_out)
        dec_out = dec_out + pos

        # connector decoder
        dec_out = self.connect_decoder(dec_out, enc_out)

        upsamplers_iter = iter(self.upsamplers)
        encs_iter = iter(encs)
        upsampler = None
        enc = None

        for i, decoder in enumerate(self.decoders):
            # at the beggining of each block...
            if i % self.decoders_per_block == 0:
                # assigning upsampler and encdoing from parallel encoding block
                enc = next(encs_iter)
                upsampler = next(upsamplers_iter)

                # using upsampler
                dec_out = upsampler(dec_out)

                # checking if we match with desired output shape
                # if we have seq_len dimension mismatch we compress oraz upscale the signal with interpolation
                mismatch = dec_out.shape[1] != enc.shape[1]
                if mismatch:
                    # interpolation operate on (B, channels, seq_len)
                    # dec_out = (B, hidden_dim, seq_len)
                    dec_out = dec_out.transpose(2, 1)
                    # now we have aligned seq_len dimension
                    dec_out = F.interpolate(dec_out, enc.shape[1], mode="linear", align_corners=False)
                    # dec_out = (B, seq_len, hidden_dim)
                    # bakc to the blocks sizes
                    dec_out = dec_out.transpose(2, 1)

                # adding position after upsampling
                pos = self.dec_pos_emb(dec_out)
                dec_out = dec_out + pos

            #print("Decoder: ", dec_out.shape, enc.shape, end="\n")

            if self.grad_checkpointing and self.training:
                dec_out = torch.utils.checkpoint.checkpoint(decoder, dec_out, enc, use_reentrant=False)
            else:
                dec_out = decoder(dec_out, enc)

        # after grind
        # dec_out = (batch_size, seq_len, hidden_dim)

        # last resort check if somehow sizes are not equal
        if dec_out.shape[1] != starting_shape[1]:
            dec_out = dec_out.transpose(2, 1)
            dec_out = F.interpolate(dec_out, starting_shape[1], mode="linear", align_corners=False)
            dec_out = dec_out.transpose(2, 1)

        # dec_out = (batch_size, seq_len, channels)
        dec_out = self.proj(dec_out)

        return dec_out

    @torch.no_grad()
    def encode(self, x):
        self.eval()
        # ----- ENCODER PASS -----
        # x = (batch_size, seq_len, hidden_dim)
        enc_out = self.encoder_embedding(x)
        # pos_enc = (batch_size, seq_len, hidden_dim)
        enc_pos_enc = self.encoder_pos_enc(x)
        # adding learned position encoding
        enc_out = enc_out + enc_pos_enc


        # x = (batch_size, seq_len/(num_encoder//2), hidden_dim)
        downsamplers_iter = iter(self.downsamplers)
        for i,encoder_block in enumerate(self.encoders):
            if self.grad_checkpointing and self.training:
                enc_out = torch.utils.checkpoint.checkpoint(encoder_block, enc_out, use_reentrant=False)
            else:
                enc_out = encoder_block(enc_out)

            if ((i + 1) % self.encoders_per_block == 0):

                # downsampling
                downsampler = next(downsamplers_iter)
                enc_out = downsampler(enc_out)

        latent = self.latent_pool(enc_out)  # (B, 128)
        latent = self.latent_proj(latent)  # (B, latent_dim)

        return latent

if __name__ == "__main__":
    config = TransformerAecConfig()
    model =TransformerAec(config=config)

    x = torch.ones(6, 60, 12)

    x_melt = model(x)

    print(x.shape)
    print(x_melt.shape)