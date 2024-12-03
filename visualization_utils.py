import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

def make_predictions(model, data_loader, device):

    model.eval()
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_outputs.extend(outputs.cpu().numpy())  
            all_labels.extend(y_batch.numpy())       
    
    return np.array(all_outputs), np.array(all_labels)

def plot_predictions(predictions, actuals, dates, scalers, set_splits, title='Stock Price Prediction'):

    plt.figure(figsize=(15, 6))

    plt.plot(dates, actuals, 'k-', label='Actual', alpha=0.3)

    colors = {
        'train': '#2ecc71',
        'val': '#f1c40f',
        'test': '#e74c3c'
    }
    
    for set_name, (start, end) in set_splits.items():
        plt.plot(dates[start:end], 
                predictions[start:end], 
                color=colors[set_name], 
                label=f'{set_name.capitalize()} Prediction')
    
    plt.title('Closing Price Prediction', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def calculate_prediction_metrics(predictions, actuals, scalers):

    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    mae = np.mean(np.abs(predictions - actuals))
    
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }


def plot_test_predictions_only(model, test_loader, dates, sequence_length, device, title='Test Set Stock Price Prediction'):

    test_pred, test_actual = make_predictions(model, test_loader, device)
    
    test_dates = dates[-(len(test_pred)+sequence_length):]
    test_dates = test_dates[sequence_length:]
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(test_dates, test_actual, 'k-', label='Actual', alpha=0.6)
    
    plt.plot(test_dates, test_pred, color='#e74c3c', label='Test Prediction', alpha=0.8)
    
    plt.title(title, fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    
    metrics = calculate_prediction_metrics(test_pred, test_actual, None)
    plt.annotate(f'RMSE: {metrics["RMSE"]:.2f}\nMAE: {metrics["MAE"]:.2f}\nMAPE: {metrics["MAPE"]:.2f}%', 
                xy=(0.02, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

    print("\nTest Set Metrics:")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    