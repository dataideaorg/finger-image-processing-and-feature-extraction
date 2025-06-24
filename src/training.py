# %%
import os
import torch
import torch.optim as optim
from .utils import create_siamese_pairs, plot_training_curves

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")    
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs") 
os.makedirs(LOGS_DIR, exist_ok=True)

# %%
def train_siamese_network(model, criterion, train_dataset, val_dataset, device, 
                         epochs=50, batch_size=32, lr=0.001):
    print("=== Training Siamese Network ===")
    
    print("Creating training pairs...")
    train_pairs, train_labels = create_siamese_pairs(train_dataset, num_pairs=2000)
    
    print("Creating validation pairs...")
    val_pairs, val_labels = create_siamese_pairs(val_dataset, num_pairs=500)
    
    train_pairs_tensor = []
    val_pairs_tensor = []
    
    for pair in train_pairs:
        train_pairs_tensor.append((pair[0].unsqueeze(0), pair[1].unsqueeze(0)))
    
    for pair in val_pairs:
        val_pairs_tensor.append((pair[0].unsqueeze(0), pair[1].unsqueeze(0)))
    
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.float32)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        train_batches = 0
        
        for i in range(0, len(train_pairs_tensor), batch_size):
            batch_pairs = train_pairs_tensor[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size].to(device)
            
            batch_x1 = torch.cat([pair[0] for pair in batch_pairs], dim=0).to(device)
            batch_x2 = torch.cat([pair[1] for pair in batch_pairs], dim=0).to(device)
            
            optimizer.zero_grad()
            
            features1, features2 = model(batch_x1, batch_x2)
            loss = criterion(features1, features2, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            train_batches += 1
        
        model.eval()
        epoch_val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_pairs_tensor), batch_size):
                batch_pairs = val_pairs_tensor[i:i+batch_size]
                batch_labels = val_labels[i:i+batch_size].to(device)

                batch_x1 = torch.cat([pair[0] for pair in batch_pairs], dim=0).to(device)
                batch_x2 = torch.cat([pair[1] for pair in batch_pairs], dim=0).to(device)
                
                features1, features2 = model(batch_x1, batch_x2)
                loss = criterion(features1, features2, batch_labels)
                
                epoch_val_loss += loss.item()
                val_batches += 1
        
        avg_train_loss = epoch_train_loss / train_batches if train_batches > 0 else 0
        avg_val_loss = epoch_val_loss / val_batches if val_batches > 0 else 0
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
    
    print("Training completed!")
    
    plot_training_curves(train_losses, val_losses)
    
    return train_losses, val_losses 