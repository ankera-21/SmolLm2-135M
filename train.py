from SmolLm3 import LlamaModel
import torch
import yaml
from transformers import AutoTokenizer
# from gptdataloader import GPTDataLoader
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset
import logging
import math

from utils import upload_file_to_s3
# At the start of training loop
# print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
# print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('training.log')
file_handler.setFormatter(formatter)  # Set formatter on the handler, not the logger
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

def encode_text(examples, tokenizer, seq_length):
    """Tokenize and prepare text examples for training."""
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=seq_length + 1,
        return_tensors="pt",
    )
    # Use clone().detach() as recommended
    input_ids = tokens["input_ids"].squeeze(0).clone().detach()
    input_ids = torch.clamp(input_ids, min=0, max=tokenizer.vocab_size - 1)
    labels = input_ids.clone().detach()
    labels = labels[1:].to(torch.int64)
    input_ids = input_ids[:-1].to(torch.int64)

    return {"input_ids": input_ids, "labels": labels}

def load_cosmopedia_dataset(batch_size=8, seq_length=1024, tokenizer=None):
    """
    Returns a torch dataloader for the cosmopedia dataset
    """
    # Set tokenizer parallelism explicitly
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logger.info("tokenizer parallelism set to false")
    try:
        # Increase timeout and retries for dataset loading
        from datasets import config
        config.HF_DATASETS_TIMEOUT = 300  # 5 minutes timeout
        config.MAX_RETRIES = 10  # Increase retry attempts
        logger.info("dataset loading config set")
        train_dataset = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            name="cosmopedia-v2",
            split="train",
            streaming=True,
        )
        logger.info("dataset loaded")

        # Use partial to bind tokenizer and seq_length to the encode function
        from functools import partial
        encode_fn = partial(encode_text, tokenizer=tokenizer, seq_length=seq_length)
        
        train_dataset = train_dataset.map(
            encode_fn, 
            remove_columns=["text"], 
            batched=False
        )
        train_dataset = train_dataset.with_format("torch")
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )
        return train_dataloader
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None
    
# def create_dataloader(file_path, tokenizer, context_size, stride):
#     with open(file_path, "r") as file:
#         text_data = file.read()
#     total_characters = len(text_data)
#     total_tokens = len(tokenizer.encode(text_data))
#     logger.info(f"Characters: {total_characters}")
#     logger.info(f"Tokens: {total_tokens}")

#     # create dataloader
#     train_ratio = 0.9
#     val_ratio = 0.1
#     split_idx  =   int(train_ratio * total_characters)

#     train_data = text_data[:split_idx]

#     valid_data = text_data[split_idx:]
    
#     train_dataset = GPTDataLoader(train_data, tokenizer, context_size, stride)
#     valid_dataset = GPTDataLoader(valid_data, tokenizer, context_size, stride)  
#     return DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=True), DataLoader(valid_dataset, batch_size=10, shuffle=False, drop_last=True)




# def calculate_loss_batch(input_batch, target_batch, model, device):
#     input_batch = input_batch.to(device)
#     target_batch = target_batch.to(device)
    
#     logits, loss = model(input_batch, target_batch) # e.g. 10, 32, 49152
#     logits = logits.view(-1, logits.size(-1))  # Shape: [320, 49152]
#     target_batch = target_batch.view(-1)  # Shape: [320]
#     loss = torch.nn.functional.cross_entropy(logits, target_batch)
#     return loss

# def calc_loss_loader(data_loader, model, device, num_batches=None):
#     total_loss = 0.0
#     if len(data_loader) == 0:
#         return float("nan")
#     elif num_batches is None:
#         num_batches = len(data_loader)
#     else:
#         num_batches = min(num_batches, len(data_loader))
#     for i, (input_batch, target_batch) in enumerate(data_loader):
#         if i < num_batches:
#             loss = calculate_loss_batch(input_batch, target_batch, model, device)
#             total_loss += loss.item()
#         else:
#             break
#     return total_loss / num_batches

# def evaluate_model(model, train_dataloader, valid_dataloader, device, eval_iter=100):
#     model.eval()
#     with torch.no_grad():
#         train_loss = calc_loss_loader(train_dataloader, model, device, num_batches=eval_iter)
#         valid_loss = calc_loss_loader(valid_dataloader, model, device, num_batches=eval_iter)
#     model.train()
#     return train_loss, valid_loss

def generate(model, idx, max_new_tokens, context_length, temperature=1.0, top_k=None, eos_token=None, device=None):
    logger.info(f"Generating on device {device}")
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

def sync_device(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    elif device == 'cpu':
        torch.cpu.synchronize() if hasattr(torch.cpu, 'synchronize') else None
    elif device.startswith('mps'):  # For Apple Silicon
        torch.mps.synchronize()

def print_gpu_memory(step_name=""):
    """
    Print GPU memory statistics with a specified step name
    """
    if torch.cuda.is_available():
        logger.info(f"\nGPU Memory Stats {step_name}:")
        logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        logger.info(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        logger.info(f"Max GPU Memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

# Learning rate scheduler
def get_lr_lambda(current_step, warmup_steps, max_steps, max_lr):
    """
    Modified learning rate scheduler with:
    1. Linear warmup for first 3000 steps
    2. Cosine decay from 3000 to 60000 steps
    3. Minimum learning rate of 1.5e-5 (5% of max_lr)
    """
    min_lr = max_lr * 0.05  # Minimum learning rate (5% of max_lr)

    if current_step < warmup_steps:
        # Linear warmup from 0 to max_lr
        return float(current_step) / float(max(1, warmup_steps))
    else:
        # Cosine decay from max_lr to min_lr
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train_model(config, model, train_loader, test_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context="Jack Gisburn rather a cheap genius- ", tokenizer=None):
    total_loss = 0
    tokens_seen, global_step = 0, -1
    
    # Adjusted gradient accumulation setup
    actual_batch_size = config['tokens']['micro_batch_size']  # Now 16
    effective_batch_size_multiplier = 2  # Reduced from 4 to maintain reasonable memory usage
    target_batch_size = effective_batch_size_multiplier * config['tokens']['micro_batch_size']
    gradient_accumulation_steps = target_batch_size // actual_batch_size
    
    # Adjusted learning rate parameters for new batch size
    max_lr = 3e-4  # Keep the same max learning rate
    warmup_steps = 3000  # Increase warmup steps for longer training
    max_steps = 60000  # Set to match 10 hours of training
    min_lr = max_lr * 0.05  # Reduce minimum LR to 5% of max (was 10%)
    
    # Create LambdaLR scheduler with the improved lambda function
    lr_lambda = lambda step: get_lr_lambda(step, warmup_steps, max_steps, max_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    logger.info(f"Training with learning rate schedule:")
    logger.info(f"Max LR: {max_lr}")
    logger.info(f"Warmup Steps: {warmup_steps}")
    logger.info(f"Max Steps: {max_steps}")
    logger.info(f"Min LR: {max_lr * 0.05}")
    logger.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
    logger.info(f"Effective Batch Size: {actual_batch_size * gradient_accumulation_steps}")
    
    print_gpu_memory("at start of training")
    
    # Add these near the start of training loop
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for batch_idx, batch in enumerate(train_loader):
            input_batch = batch['input_ids'].to(device)
            target_batch = batch['labels'].to(device)
            
            # Forward pass
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, original_loss = model(input_batch, target_batch)
            
                # Scale loss for gradient accumulation
            scaled_loss = original_loss / gradient_accumulation_steps
            scaled_loss.backward()
            
            # Add the original loss to total_loss for logging
            total_loss += original_loss.item()  # Don't multiply back up
            tokens_seen += input_batch.numel()
            
            # Calculate running average loss
            total_batches = batch_idx + 1
            avg_loss = total_loss / total_batches
            if batch_idx % 25 == 0:
                logger.info(f"Batch {batch_idx + 1}, Running Avg Loss: {avg_loss:.5f}")
            # Only update weights after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()  # Update learning rate
                optimizer.zero_grad()
                global_step += 1
            
            # Evaluation block
            if global_step % eval_freq == 0 and global_step > 0:
                # Use total batches processed instead of global_step
                current_lr = scheduler.get_last_lr()[0]
                optimizer_lr = optimizer.param_groups[0]['lr']
                
                print_gpu_memory(f"at step {global_step}")
                logger.info(f"learning rate: {current_lr:.8f}")
                logger.info(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Avg loss {avg_loss:.3f} | {tokens_seen} tokens seen")
                logger.info(f"optimizer lr: {optimizer_lr:.8f}")
                logger.info(f"scheduler lr: {current_lr:.8f}")
                
                # Generate sample text
                encoded_text = tokenizer.encode(start_context, return_tensors="pt")
                random_topk = np.random.randint(1, 10)
                logger.info(f"random_topk: {random_topk}")
                random_temperature = np.random.uniform(0.7, 0.9)
                logger.info(f"random_temperature: {random_temperature}")
                logger.info(f"global step {global_step} , batch_idx {batch_idx} => generating text")
                generated_text = generate(model, 
                                       idx=encoded_text,
                                       max_new_tokens=256,
                                       context_length=256, 
                                       temperature=random_temperature, 
                                       top_k=random_topk, 
                                       eos_token=tokenizer.eos_token_id, 
                                       device=device)
                logger.info(f"+++"*30)
                logger.info(tokenizer.decode(generated_text.squeeze(0)))
                logger.info(f"+++"*30)
                
                # Save checkpoint
                model_file_name = f"model_{global_step}_steps_avg_loss_{avg_loss:.5f}_optimizer_lr_{optimizer_lr:.8f}.pth"
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                }, model_file_name)
                
                s3_path = upload_file_to_s3(model_file_name, config['model']['model_config']['s3_bucket'], 
                                          config['model']['model_config']['s3_checkpoint_folder'])
                logger.info(f"Model saved to S3: {s3_path}")

                log_path = upload_file_to_s3(config['model']['model_config']['s3_log_file_name'], config['model']['model_config']['s3_bucket'], 
                                              config['model']['model_config']['s3_log_folder'])
                logger.info(f"Log saved to S3: {log_path}")
            
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx} finished")
                logger.info(f"+++"*30)

    logger.info("Training complete")

if __name__ == "__main__":
    config = yaml.load(open("config_smollm2_135M.yaml", "r"), Loader=yaml.FullLoader)
    logger.info(config)
    
    # Set memory efficient settings
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Empty cache before model creation
    torch.cuda.empty_cache()
    
    model = LlamaModel(config['model'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Enable gradient checkpointing for memory efficiency
    # model.gradient_checkpointing_enable()
    
    model.to(device)
    model = torch.compile(model)
    logger.info(model)
    logger.info("++"*30)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=3e-4, 
        weight_decay=0.15,
        betas=(0.9, 0.95)
    )
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    
    # Adjusted batch size and sequence length
    train_loader = load_cosmopedia_dataset(
        batch_size=16,  # Set to 16
        seq_length=1024,  # Kept at 1024
        tokenizer=tokenizer
    )
    
    import time
    t1 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set environment variable for memory allocation
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    train_model(
        config, 
        model, 
        train_loader, 
        train_loader, 
        optimizer=optimizer, 
        device=device, 
        num_epochs=1, 
        eval_freq=1000,  # Increase eval frequency to every 500 steps
        eval_iter=1000,
        start_context="Once Upon a Time far far away in a galaxy", 
        tokenizer=tokenizer
    )
    t2 = time.time()
    logger.info(f"Time taken for training: {t2 - t1:.2f} seconds")