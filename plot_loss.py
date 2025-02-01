import re
import matplotlib.pyplot as plt
import numpy as np

# Read the log file
with open('training.log', 'r') as file:
    log_content = file.read()

# Regular expression pattern to match batch number and loss
pattern = r'Batch (\d+), Running Avg Loss: ([\d.]+)'

batches = []
losses = []

# Extract data from the log content
for line in log_content.split('\n'):
    match = re.search(pattern, line)
    if match:
        batch_num = int(match.group(1))
        loss = float(match.group(2))
        batches.append(batch_num)
        losses.append(loss)

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(batches, losses)
plt.title('Training Loss over Batches')
plt.xlabel('Step Number')
plt.ylabel('Running Average Loss')
plt.grid(True)

# Add trend line
z = np.polyfit(batches, losses, 1)
p = np.poly1d(z)
plt.plot(batches, p(batches), "r--", alpha=0.8, label=f'Trend line')

plt.legend()
plt.tight_layout()
plt.savefig('training_loss_plot.png')
plt.show()

# Print some statistics
print(f"Number of data points: {len(batches)}")
print(f"Initial loss: {losses[0]:.5f}")
print(f"Final loss: {losses[-1]:.5f}")
print(f"Loss reduction: {losses[0] - losses[-1]:.5f}")
