from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        is_multiheaded = not (heads == 1 and dim_head == dim)

        self.heads = heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.scale = dim_head ** -0.5 # = 1 / sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if is_multiheaded else nn.Identity()
        
    def scaled_dot_prod_attn(self, q, k, v):
        scaled_qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(scaled_qk)
        attn = self.dropout(attn)
        attn = torch.matmul(attn, v)

        return attn

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        attn = self.scaled_dot_prod_attn(q, k, v)
        attn = rearrange(attn, 'b h n d -> b n (h d)')

        return self.out(attn)

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Dropout(dropout)
            nn.Linear(dim*4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])

        self.attn_mask = None

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads), # From pytorch. TODO: Fix custom one
                nn.LayerNorm(dim),
                MLP(dim, mlp_dim, dropout)
            ]))
            
    def attention(self, x: torch.Tensor, attn_layer):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return attn_layer(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def forward(self, x):
        for mh_attn_norm, mh_attn, mlp_norm, mlp in self.layers:
            x_ = mh_attn_norm(x)
            x = x + mh_attn(x_, x_, x_)[0]
            x = x + mlp(mlp_norm(x))

        return x


class ViT(nn.Module):
    def __init__(self,
         transformer_depth: int,
         attn_heads: int,
         mlp_dim: int,
         output_dim: int,
         dim_attn_head: int = 64,
         channels: int = 3,
         patch_width: int = 16,
         patch_height: int = 16,
         image_width: int = 16,
         image_height: int = 16,
         patch_embeddings_dim: int = 1024,
         emb_dropout: float = 0.0,
         dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.patch_dim = channels * patch_height * patch_width
        self.patch_embeddings_dim = patch_embeddings_dim
        num_patches = (image_width // patch_width) * (image_height // patch_height)
        patch_size = patch_width
        self.scale = dim_attn_head ** -0.5 # = 1 / sqrt(d_k)
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=patch_embeddings_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.cls_token = nn.Parameter(self.scale * torch.randn(patch_embeddings_dim))
        self.positional_embedding = nn.Parameter(self.scale * torch.randn(num_patches + 1, patch_embeddings_dim))
        
        self.dropout = nn.Dropout(emb_dropout)

        self.ln_pre = nn.LayerNorm(patch_embeddings_dim)
        self.trasnformer = Transformer(patch_embeddings_dim, transformer_depth, attn_heads, dim_attn_head, mlp_dim, dropout)
        self.ln_post = nn.LayerNorm(patch_embeddings_dim)
        
        self.out_proj = nn.Parameter(self.scale * torch.randn(patch_embeddings_dim, output_dim))
        
    def embed_image(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.cls_token.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.dropout(x)
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.trasnformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        x = self.ln_post(x[:, 0, :])

        return x

    def forward(self, x):
        return self.embed_image(x)
