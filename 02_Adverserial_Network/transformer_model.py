import torch
import torch.nn as nn
from flash_attn import flash_attn_func  # Install on server: pip install flash-attn

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
        attn_output = flash_attn_func(q, k, v, dropout_p=0.0)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(attn_output)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=4, embed_dim=128, num_heads=8, num_layers=4, num_classes=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList([
            FlashAttentionLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)  # [B, N, embed_dim]
        for layer in self.layers:
            x = layer(x) + x  # Residual connection
        x = x.mean(dim=1)  # Pool over sequence
        return self.classifier(x)

# Example usage
if __name__ == "__main__":
    model = TransformerClassifier()
    dummy_input = torch.randn(4, 100, 4)  # Batch of 4, seq_len=100, features=4
    output = model(dummy_input)
    print(output.shape)  # Should be [4, 2]