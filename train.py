
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from data.dataloader import get_dataloader
from models.transformer import Transformer
from utils.config import Config
from utils.logger import get_logger
from utils.utils import save_checkpoint

def train(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    for batch in dataloader:
        src = batch.to(device)
        
        optimizer.zero_grad()
        output = model(src, src[:, :-1], torch.randperm(src.size(1)-1, device=device))
        loss = criterion(output.contiguous().view(-1, output.size(-1)), src[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    wandb.log({"train_loss": avg_loss}, step=epoch)
    return avg_loss

def main():
    wandb.init(project="protein_transformer", entity="your_wandb_username")
    
    config = Config()
    wandb.config.update(config.__dict__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Transformer(config).to(device)
    dataloader = get_dataloader(config, data_file='data/uniprotkb_AND_model_organism_9606_AND_l_2024_06_17.fasta')
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx)
    
    logger = get_logger("train_log")
    
    for epoch in range(config.num_epochs):
        logger.info(f"Epoch: {epoch+1}")
        
        train_loss = train(model, dataloader, optimizer, criterion, device, epoch)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        save_checkpoint(model, optimizer, epoch, train_loss, f"checkpoints/checkpoint_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()