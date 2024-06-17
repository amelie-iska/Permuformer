
import torch
import torch.nn as nn
import wandb
from data.dataloader import get_dataloader
from models.transformer import Transformer
from utils.config import Config
from utils.logger import get_logger
from utils.utils import load_checkpoint

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            src = batch.to(device)
            
            output = model(src, src[:, :-1], torch.randperm(src.size(1)-1, device=device))
            loss = criterion(output.contiguous().view(-1, output.size(-1)), src[:, 1:].contiguous().view(-1))
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main():
    wandb.init(project="permuformer", entity="your_wandb_username")
    
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Transformer(config).to(device)
    dataloader = get_dataloader(config, data_file='data/uniprotkb_AND_model_organism_9606_AND_l_2024_06_17.fasta', shuffle=False)
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx)
    
    logger = get_logger("eval_log")
    
    checkpoint_path = "checkpoints/checkpoint_epoch_1.pt"
    model, _, _, _ = load_checkpoint(checkpoint_path, model, None)
    
    eval_loss = evaluate(model, dataloader, criterion, device)
    wandb.log({"eval_loss": eval_loss})
    logger.info(f"Evaluation Loss: {eval_loss:.4f}")

if __name__ == "__main__":
    main()