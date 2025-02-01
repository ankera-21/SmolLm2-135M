import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import SiLU
import yaml


def _init_weights(module, std=0.041666666666666664):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)

class RotaryPositionalEmbedding(nn.Module):
    """
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L240
    Rotary Positional Embedding (RoPE) for transformers Implemntation derived from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    """
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply rotary positional embedding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape # B, T, H, D
            seq_len (int): Sequence length. #T

        Returns:
            torch.Tensor: Output tensor with rotary positional embeddings applied.
        """
        B, T, H, H_D = x.shape

        # Generate position indices
        position = torch.arange(T, dtype=torch.float32, device=x.device).unsqueeze(-1)

        # Generate frequencies
        freqs = torch.exp(
            torch.arange(0, H_D, 2, dtype=torch.float32, device=x.device) * 
            -(torch.log(torch.tensor(self.theta)) / H_D)
                                                                
        )

        # Compute sinusoids
        sinusoid = position * freqs
        sin = torch.sin(sinusoid)
        cos = torch.cos(sinusoid)

        # Reshape sin and cos to match the input tensor's shape
        sin = sin.unsqueeze(0).unsqueeze(2)  # Shape: (1, T, 1, D // 2)
        cos = cos.unsqueeze(0).unsqueeze(2)  # Shape: (1, T, 1, D // 2)

        # Apply rotary embeddings
        x_rotated = x.clone()
        x_rotated[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
        x_rotated[..., 1::2] = x[..., 1::2] * cos + x[..., 0::2] * sin

        return x_rotated
    
class LlamaAttention(nn.Module):
    """
    (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=576, out_features=576, bias=False)
          (k_proj): Linear(in_features=576, out_features=192, bias=False)
          (v_proj): Linear(in_features=576, out_features=192, bias=False)
          (o_proj): Linear(in_features=576, out_features=576, bias=False)
    )
    """
    def __init__(self, config, rotary_emb):
        super().__init__()
        self.config = config
        self.num_attention_heads = self.config['num_attention_heads']
        self.hidden_size = self.config['hidden_size']
        # Ensure the hidden size is divisible by the number of attention heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
            )
        self.num_key_value_heads = self.config['num_key_value_heads']
        self.head_dim =  self.hidden_size // self.num_attention_heads
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  # D,D
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim*self.num_key_value_heads, bias=False)   # D,D/H
        self.v_proj = nn.Linear(self.hidden_size, self.head_dim*self.num_key_value_heads, bias=False)   # D,D/H
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)   # D,D

        # Convert the mask to boolean type when creating it
        # self.register_buffer("mask", 
        #                    torch.triu(torch.ones(self.config['max_position_embeddings'], 
        #                                        self.config['max_position_embeddings']),
        #                             diagonal=1))  # Convert to boolean
        
        self.rotary_pos_emb = rotary_emb

    def forward(self, x):
        B, T, C = x.size()

        q = self.q_proj(x)  # B,T,D
        k = self.k_proj(x)  # B,T,D/H
        v = self.v_proj(x)  # B,T,D/H

        q = q.view(B, T, self.num_attention_heads, self.head_dim) # B,T,H,D
        k = k.view(B, T, self.num_key_value_heads, self.head_dim) # B,T,H,D
        v = v.view(B, T, self.num_key_value_heads, self.head_dim) # B,T,H,D

        q = q.transpose(1,2) # B,H,T,D
        k = k.transpose(1,2) # B,num_key_value_heads,T,D
        v = v.transpose(1,2) # B,num_key_value_heads,T,D

        # apply rotary positional embedding
        q = self.rotary_pos_emb(q, T)
        k = self.rotary_pos_emb(k, T)

        # Repeat k/v heads if num_key_value_heads < num_attention_heads
        if self.num_key_value_heads != self.num_attention_heads:
            k = k.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1) # B,kv_head,T,D -> B,H,T,D
            v = v.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1) # B,kv_head,T,D -> B,H,T,D

        # Manual attention Stats
        # Q(B,H,T,D) @K.T(B,H,D,T) = Q.K_T (B,H,T,T)
        # attn_scores = q @ k.transpose(-2,-1) # B,H,T,T
        # mask_bool = self.mask[:T,:T].bool() # T,T
        # attn_scores.masked_fill_(mask_bool, -torch.inf) # B,H,T,T
        # attn_weights = F.softmax(attn_scores/k.size(-1)**0.5, dim=-1) # B,H,T,T
        # context_vector = attn_weights @ v # B,H,T,T * B,H,T,D = B,H,T,D
        # context_vector = context_vector.transpose(1,2) # B,T,H,D
        # context_vector = context_vector.contiguous().view(B,T,C) # B,T,H,D -> B,T,D
        # Manual attention Stats ENDS

        # Scaled dot-product attention STARTS   
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        context_vector = attn_out.transpose(1,2).reshape(B,T,C)
        # Scaled dot-product attention ENDS

        context_vector = self.o_proj(context_vector)
        
        return context_vector


class LlamaMLP(nn.Module):
    """
    (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
          (up_proj): Linear(in_features=576, out_features=1536, bias=False)
          (down_proj): Linear(in_features=1536, out_features=576, bias=False)
          (act_fn): SiLU()
        )
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(self.config['hidden_size'], self.config['intermediate_size'], bias=False)
        self.up_proj = nn.Linear(self.config['hidden_size'], self.config['intermediate_size'], bias=False)
        self.down_proj = nn.Linear(self.config['intermediate_size'], self.config['hidden_size'], bias=False)
        self.act_fn = SiLU()
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        down = self.down_proj(self.act_fn(gate)*up)
        return down 
    
class LlamaRMSNorm(nn.Module):
    """
    (norm): LlamaRMSNorm((576,), eps=1e-05)
        # RMSNorm Formula:
        #    RMS(x) = sqrt((1 / d) * sum(x_i^2 for i in range(d)))
        #    x_normalized = x / RMS(x)
        #    output = gamma * x_normalized
    
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.eps = self.config['rms_norm_eps']
        self.weight = nn.Parameter(torch.ones(self.config['hidden_size']))
    def forward(self, x):
        rms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return  self.weight *rms * x
    
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, rotary_emb):
        super().__init__()
        self.config = config
        self.self_attn = LlamaAttention(self.config, rotary_emb)
        self.mlp = LlamaMLP(self.config)
        self.input_layernorm = LlamaRMSNorm(self.config)
        self.post_attention_layernorm = LlamaRMSNorm(self.config)   
    
    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = x + residual

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual
        return x 
    
class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.init_method = config['init_method']
        self.config = config['model_config']
        self.embed_tokens = nn.Embedding(self.config['vocab_size'], self.config['hidden_size'])
        self.rotary_emb = RotaryPositionalEmbedding(self.config['hidden_size'], self.config['rope_theta'])
        self.layers = nn.ModuleList([LlamaDecoderLayer(self.config, self.rotary_emb) for _ in range(self.config['num_hidden_layers'])])
        self.norm = LlamaRMSNorm(self.config)
        self.lm_head = nn.Linear(self.config['hidden_size'], self.config['vocab_size'], bias=False)
        
        if self.config['tie_word_embeddings']:
            self.lm_head.weight = self.embed_tokens.weight
        
        self.apply(lambda m: _init_weights(m, self.init_method['std']))
    
    def forward(self, x, y=None):
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x) # B,T,V
        logits = logits.view(-1, logits.size(-1))  # Shape: [B*T, V]
        if y is not None:
            y = y.view(-1)  # Shape: [B*T]
            loss = torch.nn.functional.cross_entropy(logits, y)
            return logits, loss
        else:
            return logits, None

    
    def generate(self, idx, max_new_tokens, context_length, temperature=1.0, top_k=None, eos_token=None, device=None):
        model = self.to(device)
        idx = idx.to(device)
        model.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_length:]
            with torch.no_grad():
                logits, _ = model(idx_cond)  # Unpack both logits and loss (ignore loss)
                logits = logits.view(idx_cond.shape[0], -1, model.config['vocab_size'])  # Reshape to [batch, seq, vocab]
                
            # Get the logits for the last token only
            logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
            
            if top_k is not None:
                # top k sampling
                top_logits, top_pos = torch.topk(logits, top_k)
                min_logit = top_logits[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_logit,
                                torch.tensor(float('-inf')).to(logits.device),
                                logits)
            
            # temperature scaling
            if temperature > 0.0:
                logits /= temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                
            if idx_next.item() == eos_token:
                break
                
            idx = torch.cat((idx, idx_next), dim=1)
        model.train()
        return idx

# if __name__ == "__main__":
#     torch.manual_seed(0)
#     config = yaml.load(open("config_smollm2_135M.yaml", "r"), Loader=yaml.FullLoader)
#     print(config.keys())
#     model_config = config['model']['model_config']
#     print(model_config)
#     model = LlamaModel(config['model'])
#     x_tokens = torch.randint(0, model_config['vocab_size'], (1, 10))  # Generate random token indices
#     print(model(x_tokens).shape)
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Total parameters: {total_params}") #134515008
#     print(model)