import argparse
from SmolLm3 import LlamaModel
import yaml
import torch
from transformers import AutoTokenizer


def generate_helper(model, idx, max_new_tokens, context_length, temperature=1.0, top_k=None, eos_token=None, device=None):
    
    model = model.to(device)
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


def get_config(config_path):
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    return config

def load_weights(config, weights_path, device):
    model = LlamaModel(config['model'])
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    return model

def load_model_from_checkpoint(config_path, checkpoint_path, device):
    config = get_config(config_path)
    model = LlamaModel(config['model'])
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model

def get_tokenizer(config):
    tokenizer_path = config['tokenizer']['tokenizer_name_or_path']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    return tokenizer, vocab_size

def generate_text(model, tokenizer, input_text, max_new_tokens, context_length, temperature, top_k, eos_token, device):
    encoded_text = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    generated_text = generate_helper(model, 
                            idx=encoded_text,
                            max_new_tokens=max_new_tokens,
                            context_length=context_length, 
                            temperature=temperature, 
                            top_k=top_k, 
                            eos_token=eos_token, 
                            device=device)
    return tokenizer.decode(generated_text.squeeze(0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text using the SmolLM model')
    parser.add_argument('--config_path', type=str, default="config_smollm2_135M.yaml",
                        help='Path to the config file')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--input_text', type=str, default="Once upon a time in far far away",
                        help='Input text prompt for generation')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--context_length', type=int, default=256,
                        help='Context length for generation')
    parser.add_argument('--temperature', type=float, default=0.9,
                        help='Temperature for sampling')
    parser.add_argument('--top_k', type=int, default=2,
                        help='Top-k value for sampling')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='Device to run the model on (cuda/cpu)')
    
    args = parser.parse_args()

    config = get_config(args.config_path)
    model = load_weights(config, args.checkpoint_path, args.device)
    print(model)
    tokenizer, vocab_size = get_tokenizer(config)
    print(tokenizer)
    print(vocab_size)

    generated_text = generate_text(
        model, 
        tokenizer, 
        args.input_text,
        args.max_new_tokens,
        args.context_length,
        args.temperature,
        args.top_k,
        tokenizer.eos_token_id,
        args.device
    )
    print(generated_text)
    print("--------------------------------")
    encoded_text = tokenizer.encode(args.input_text, return_tensors="pt").to(args.device)
    print(encoded_text)
    generated_text2=model.generate(idx=encoded_text, max_new_tokens=args.max_new_tokens, context_length=args.context_length, temperature=args.temperature, top_k=args.top_k, eos_token=tokenizer.eos_token_id, device=args.device)
    decoded_text2=tokenizer.decode(generated_text2.squeeze(0))
    print(decoded_text2)