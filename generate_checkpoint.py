import torch
from train import LlamaModel
from model_testing import get_config

config_path = "config_smollm2_135M.yaml"
config = get_config(config_path)
model = LlamaModel(config['model'])

checkpoint_path = "model_37000_steps_avg_loss_2.85920_optimizer_lr_0.00000003.pth"
weights_save_path = "model_weights_37000step1.pt"
from  model_testing import load_model_from_checkpoint
model = load_model_from_checkpoint(config_path, checkpoint_path, "cpu")
torch.save(model.state_dict(), weights_save_path)
print(f"Model weights saved to: {weights_save_path}")
model.load_state_dict(torch.load(weights_save_path, map_location=torch.device("cpu")))
print(model)
