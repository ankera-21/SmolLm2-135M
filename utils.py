import boto3
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm
import os

def upload_file_to_s3(file_path, bucket_name, s3_prefix):
    

    class ProgressPercentage(object):
        def __init__(self, filename):
            self._filename = filename
            self._size = float(os.path.getsize(filename))
            self._seen_so_far = 0
            self._pbar = tqdm(total=self._size, unit='B', unit_scale=True, desc=f"Uploading {os.path.basename(filename)}")

        def __call__(self, bytes_amount):
            self._seen_so_far += bytes_amount
            self._pbar.update(bytes_amount)

    s3_client = boto3.client('s3')
    file_name = os.path.basename(file_path)
    s3_path = f"{s3_prefix}/{file_name}"
    
    # Configure multipart upload
    config = TransferConfig(
        multipart_threshold=1024 * 25,  # 25MB
        max_concurrency=10,
        multipart_chunksize=1024 * 25,  # 25MB
        use_threads=True
    )
    
    try:
        s3_client.upload_file(
            file_path, 
            bucket_name, 
            s3_path,
            Config=config,
            Callback=ProgressPercentage(file_path)
        )
        return f"s3://{bucket_name}/{s3_path}"
    except Exception as e:
        print(f"Failed to upload {file_path} to S3: {str(e)}")
        return None
    
max_lr = 1e-3
warmup_steps = 10
max_steps = 25000
import math
def get_lr_lambda(current_step, warmup_steps, max_steps, max_lr):
    """
    Learning rate scheduler with:
    1. Linear warmup
    2. Cosine decay
    3. Minimum learning rate of 10% of max_lr
    """
    min_lr = max_lr * 0.1  # Minimum learning rate (10% of max_lr)
    
    if current_step < warmup_steps:
        # Linear warmup
        return max_lr * (current_step + 1) / warmup_steps
    elif current_step > max_steps:
        # After max_steps, return minimum learning rate
        return min_lr
    else:
        # Cosine decay between warmup_steps and max_steps
        decay_ratio = (current_step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


def plot_lr_schedule():
    """
    Helper function to visualize the learning rate schedule
    """
    import matplotlib.pyplot as plt
    steps = list(range(0, max_steps + 100))
    lrs = [get_lr_lambda(step, warmup_steps, max_steps, max_lr) for step in steps]
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_lr_schedule()