from pp import load_and_preprocess
from dataset import GestureDataset
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from enhanced_model import EnhancedGestureCNN, LSTMGestureModel, ImprovedCNNModel
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def get_class_weights(y_train):
    """Calculate class weights for imbalanced datasets"""
    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    weights = {}
    for cls, count in zip(unique, counts):
        weights[cls] = total / (len(unique) * count)
    return weights

def train_model(model, train_loader, test_loader, device, epochs=50):
    """Enhanced training function with better monitoring"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    train_losses = []
    test_accuracies = []
    best_accuracy = 0
    
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
            torch.save(model.state_dict(), "best_gesture_model.pth")
        
        train_losses.append(avg_loss)
        test_accuracies.append(test_acc)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            print(f"Best Test Acc: {best_accuracy:.2f}%")
            print("-" * 50)
    
    return train_losses, test_accuracies, best_accuracy

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

def main():
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
        "Enhanced CNN": EnhancedGestureCNN(input_size=6, num_classes=len(le.classes_)),
        "Improved CNN": ImprovedCNNModel(input_size=6, num_classes=len(le.classes_)),
        "LSTM": LSTMGestureModel(input_size=6, num_classes=len(le.classes_))
    }
    
    results = {}
    
    for model_name, model in models_to_try.items():
        print(f"\nTraining {model_name}...")
        print("=" * 60)
        
        model = model.to(device)
        train_losses, test_accuracies, best_accuracy = train_model(
            model, train_loader, test_loader, device, epochs=30
        )
        
        # Load best model for evaluation
        model.load_state_dict(torch.load("best_gesture_model.pth"))
        pred_labels, actual_labels = evaluate_model(model, test_loader, le, device)
        
        results[model_name] = {
            'best_accuracy': best_accuracy,
            'train_losses': train_losses,
            'test_accuracies': test_accuracies
        }
        
        # Save model with specific name
        torch.save(model.state_dict(), f"{model_name.lower().replace(' ', '_')}_gesture_model.pth")
    
    # Print comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON:")
    print("="*60)
    for model_name, result in results.items():
        print(f"{model_name}: {result['best_accuracy']:.2f}%")

if __name__ == "__main__":
    main()