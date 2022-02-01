import torch
import torch.nn as nn
from .embed import PatchEmbed, RocketEmbed, classic_pos_encoding
from .transformer_base import Add, TransformerBlock, IndexSelect, Mean, trunc_normal_

class MVTSTransformer(nn.Module):
    def __init__(self, mvts_length: int, patch_size: int, num_channels: int, num_classes: int,
                 dim: int, depth: int, heads: int, mlp_dim: int, pool='cls', dim_head=64, dropout=0.,
                 emb_dropout=0., qkv_bias=False, in_embedding='linear', pos_encoding='simple'):
        super().__init__()

        # check that the input is ok and get the number of patches and patch dimension
        assert mvts_length % patch_size == 0, 'MVTS dimensions must be divisible by the patch size.'
        num_patches = mvts_length // patch_size

        # window the time-series into patches, and embed them
        self.ie_type = in_embedding
        assert self.ie_type in {'linear', 'rocket'}, 'Input embedding type must be either linear or rocket'
        if self.ie_type == 'linear':
            self.patch_embed = PatchEmbed(dim, patch_size, num_channels)
        elif self.ie_type == 'rocket':
            self.patch_embed = RocketEmbed(dim, patch_size, num_channels)

        # define the positional encoding
        self.pe_type = pos_encoding
        assert self.pe_type in {'simple', 'classic', 'none'}, 'Positional embedding type must be either simple, classic or none'
        if self.pe_type == 'simple':
            self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        elif self.pe_type == 'classic':
            pe = classic_pos_encoding(dim, num_patches + 1)
            self.register_buffer('pos_embedding', pe, persistent=False)
        elif self.pe_type == 'none':
            pass

        # define the add op for embedding
        if self.pe_type != 'none':
            self.add = Add()

        # define the learnable class tokens
        self.cls_tokens = nn.Parameter(torch.zeros(1, 1, dim))

        # embedding dropout and transformer architecture init
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, qkv_bias=qkv_bias,
                drop=dropout, attn_drop=dropout) for i in range(depth)])

        # set the pooling operation before the output mlp
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        if pool == 'cls':
            self.pool = IndexSelect(dim=1, indices=torch.tensor(0))
        else:
            self.pool = Mean(dim=1)

        # define the output head
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes))

        # initialization of the weights and parameters
        if self.pe_type == 'simple':
            trunc_normal_(self.pos_embedding, std=.02)
        trunc_normal_(self.cls_tokens, std=.02)
        self.apply(self._init_weights)

        # output layer and loss attributes
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, mvts):
        # get the learnable patch embedding
        x = self.patch_embed(mvts)

        # get the batch size and number of patches
        b, n, _ = x.shape

        cls_tokens = self.cls_tokens.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pe_type != 'none':
            x = self.add([x, self.pos_embedding])

        x = self.emb_dropout(x)

        # run through the transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # apply the pooling operation and the output mlp
        x = self.pool(x).squeeze(1)
        x = self.head(x)

        # apply the logsoftmax layer to obtain class likelihoods
        x = self.logsoftmax(x)

        return x

    def compute_loss(self, model_output, labels):
        # compute the negative log likelihood loss
        return  nn.functional.nll_loss(model_output, labels)

