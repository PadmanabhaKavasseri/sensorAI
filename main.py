from util.pp import load_and_preprocess
from util.dataset import GestureDataset
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from model_defs.model_defs import GestureRecCNN_V3, LSTMGestureModel, GestureRecCNN_V2, GestureRecCNN_V1, CNNLSTMModel 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define paths
RESULTS = Path("results")
MODELS = RESULTS / "models"
MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

def get_class_weights(y_train):
    """Calculate class weights for imbalanced datasets"""
    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    weights = {}
    for cls, count in zip(unique, counts):
        weights[cls] = total / (len(unique) * count)
    return weights

def train_model(model, train_loader, test_loader, device, model_name, epochs=50):
    """Enhanced training function with better monitoring"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    train_losses = []
    test_accuracies = []
    best_accuracy = 0
    best_model_path = MODELS / f"best_{model_name.lower().replace(' ', '_').replace('-', '_')}_model.pth"
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += y_batch.size(0)
            correct_train += (predicted == y_batch).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # Evaluation phase
        model.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_test += y_batch.size(0)
                correct_test += (predicted == y_batch).sum().item()
        
        test_acc = 100 * correct_test / total_test
        avg_test_loss = test_loss / len(test_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_test_loss)
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), best_model_path)
        
        train_losses.append(avg_loss)
        test_accuracies.append(test_acc)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            print(f"Best Test Acc: {best_accuracy:.2f}%")
            print("-" * 50)
    
    return train_losses, test_accuracies, best_accuracy, best_model_path

def evaluate_model(model, test_loader, le, device):
    """Detailed evaluation with confusion matrix and classification report"""
    model.eval()
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device).float()
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_actuals.extend(y_batch.numpy())
    
    # Convert back to string labels
    pred_labels = le.inverse_transform(all_predictions)
    actual_labels = le.inverse_transform(all_actuals)
    
    print("\nDetailed Evaluation:")
    print("=" * 50)
    print(classification_report(actual_labels, pred_labels))
    print("\nConfusion Matrix:")
    print(confusion_matrix(actual_labels, pred_labels))
    
    return pred_labels, actual_labels

def save_training_plots(results, model_name):
    """Save training loss and accuracy plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training losses
    ax1.plot(results['train_losses'], label='Training Loss')
    ax1.set_title(f'{model_name} - Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot test accuracies
    ax2.plot(results['test_accuracies'], label='Test Accuracy', color='orange')
    ax2.set_title(f'{model_name} - Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    # Save plots in results directory instead of models directory
    plot_path = RESULTS / f"{model_name.lower().replace(' ', '_').replace('-', '_')}_training_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training plots saved to: {plot_path}")

def main():
    print(f"Models will be saved to: {MODELS}")
    print(f"Plots and results will be saved to: {RESULTS}")
    
    # Load and preprocess data
    (X_train, y_train), (X_test, y_test), le = load_and_preprocess()
    
    # Display class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print("Class distribution in training data:")
    for cls, count in zip(le.classes_[unique], counts):
        print(f"  {cls}: {count} samples")
    
    # Create datasets
    train_dataset = GestureDataset(X_train, y_train)
    test_dataset = GestureDataset(X_test, y_test)
    
    # Create weighted sampler for balanced training
    class_weights = get_class_weights(y_train)
    sample_weights = [class_weights[label] for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    print(f"Final training data shape: {X_train.shape}")
    print(f"Final test data shape: {X_test.shape}")
    print(f"Number of classes: {len(le.classes_)}")
    print(f"Classes: {le.classes_}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Try different models
    models_to_try = {
        "GestureRecCNN_V3": GestureRecCNN_V3(input_size=6, num_classes=len(le.classes_)),
        "GestureRecCNN_V2": GestureRecCNN_V2(input_size=6, num_classes=len(le.classes_)),
        "GestureRecCNN_V1": GestureRecCNN_V1(input_size=6, num_classes=len(le.classes_)),
        "LSTM": LSTMGestureModel(input_size=6, num_classes=len(le.classes_)),
        "CNN-LSTM": CNNLSTMModel(input_size=6, num_classes=len(le.classes_))
    }
    
    results = {}
    
    for model_name, model in models_to_try.items():
        print(f"\nTraining {model_name}...")
        print("=" * 60)
        
        model = model.to(device)
        train_losses, test_accuracies, best_accuracy, best_model_path = train_model(
            model, train_loader, test_loader, device, model_name, epochs=30
        )
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(best_model_path))
        pred_labels, actual_labels = evaluate_model(model, test_loader, le, device)
        
        results[model_name] = {
            'best_accuracy': best_accuracy,
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'model_path': best_model_path
        }
        
        # Save final model with specific name (still in models directory)
        final_model_path = MODELS / f"{model_name.lower().replace(' ', '_').replace('-', '_')}_final_model.pth"
        torch.save(model.state_dict(), final_model_path)
        
        # Save training plots (now in results directory)
        save_training_plots(results[model_name], model_name)
        
        print(f"Best model saved to: {best_model_path}")
        print(f"Final model saved to: {final_model_path}")
    
    # Print comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON:")
    print("="*60)
    for model_name, result in results.items():
        print(f"{model_name}: {result['best_accuracy']:.2f}%")
    
    # Save comparison results (now in results directory instead of models directory)
    comparison_path = RESULTS / "model_comparison.txt"
    with open(comparison_path, 'w') as f:
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("="*50 + "\n\n")
        for model_name, result in results.items():
            f.write(f"{model_name}: {result['best_accuracy']:.2f}%\n")
            f.write(f"  Model path: {result['model_path']}\n\n")
    
    print(f"\nComparison results saved to: {comparison_path}")
    print(f"Models saved in: {MODELS}")
    print(f"Plots and comparison results saved in: {RESULTS}")

if __name__ == "__main__":
    main()