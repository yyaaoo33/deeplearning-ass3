import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs,model_name, device):
    model.to(device)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        val_loss = validate_model(model, val_loader, criterion, device)
        save_path = f"{model_name}_best_model.pth"
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, save_path)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            
    return val_loss / len(val_loader)

def evaluate_metrics(model, test_loader, criterion, scalers, device):
    model.eval()
    mse, mae = 0, 0
    total_samples = 0
    predictions, actuals = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            mse += criterion(outputs, y_batch).item()
            mae += torch.mean(torch.abs(outputs - y_batch)).item()

            outputs_np = outputs.cpu().numpy()
            y_batch_np = y_batch.cpu().numpy()
            
            # Modified: Only handle Close price
            pred_close = scalers['Close'].inverse_transform(outputs_np.reshape(-1, 1))
            actual_close = scalers['Close'].inverse_transform(y_batch_np.reshape(-1, 1))

            predictions.append(pred_close)
            actuals.append(actual_close)
            total_samples += outputs.size(0)
    
    mse /= len(test_loader)
    mae /= len(test_loader)
    
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    
    print(f"\nEvaluation Metrics:")
    print(f"MSE (normalized): {mse:.4f}")
    print(f"MAE (normalized): {mae:.4f}")
    
    mse_orig = np.mean((predictions - actuals) ** 2)
    mae_orig = np.mean(np.abs(predictions - actuals))
    print(f"MSE (original scale): {mse_orig:.4f}")
    print(f"MAE (original scale): {mae_orig:.4f}")
    
    print("\nValue Ranges:")
    print(f"Predictions - Min: {predictions.min():.2f}, Max: {predictions.max():.2f}")
    print(f"Actuals - Min: {actuals.min():.2f}, Max: {actuals.max():.2f}")
    
    return predictions, actuals, {
        'mse': mse,
        'mae': mae,
        'mse_orig': mse_orig,
        'mae_orig': mae_orig
    }
    