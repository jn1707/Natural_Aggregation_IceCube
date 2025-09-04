import torch
import torch.nn as nn
from flash_attn import flash_attn_func
import yaml
from pathlib import Path

# Load configuration
config_path = Path('02_Adverserial_Network/250902-config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
MODEL_PARAMS = config.get('MODEL_PARAMS')
EMBED_DIM = MODEL_PARAMS.get('EMBED_DIM')
NUM_HEADS = MODEL_PARAMS.get('NUM_HEADS')
NUM_LAYERS = MODEL_PARAMS.get('NUM_LAYERS')
DROPOUT = MODEL_PARAMS.get('DROPOUT')
MAX_SEQ_LENGTH = config['DATA']['MAX_SEQ_LENGTH']

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=MAX_SEQ_LENGTH):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        # x: [B, N, embed_dim]
        B, N, _ = x.shape
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)
        return self.pos_embedding(positions)

class FlashAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = FlashAttentionLayer(embed_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None):
        # LayerNorm before attention
        x_norm = self.norm1(x)
        # FlashAttention (masking not directly supported in flash_attn_func, so mask input if needed)
        attn_out = self.attn(x_norm)
        x = x + self.dropout(attn_out)  # Residual connection
        # LayerNorm before FFN
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + ffn_out  # Residual connection
        return x

class FlashAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_output = flash_attn_func(q, k, v, dropout_p=DROPOUT)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(attn_output)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=4, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, num_classes=2, max_seq_length=MAX_SEQ_LENGTH, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_length)
        self.layers = nn.ModuleList([
            FlashAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attention_mask=None):
        # x: [B, N, input_dim]
        x = self.embedding(x)  # [B, N, embed_dim]
        x = x + self.pos_encoding(x)  # Add positional encoding

        for layer in self.layers:
            x = layer(x, attn_mask=attention_mask)

        x = self.norm_final(x)
        x = x.mean(dim=1)  # Pool over sequence
        logits = self.classifier(x)
        probs = self.softmax(logits)
        return probs

# Example usage
if __name__ == "__main__":
    model = TransformerClassifier()
    dummy_input = torch.randn(4, 100, 4)  # Batch of 4, seq_len=100, features=4
    output = model(dummy_input)
    print(output.shape)  # Should be [4, 2]