from pp import load_and_preprocess
from dataset import GestureDataset
import torch
from torch.utils.data import DataLoader
from model_defs.model_defs import GestureRecCNN_V3, LSTMGestureModel, GestureRecCNN_V2, GestureRecCNN_V1, CNNLSTMModel 
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import os

class GestureModelEvaluator:
    def __init__(self):
        """Initialize the evaluator with absolute paths."""
        # Use absolute paths to avoid path resolution issues
        self.models_path = Path("/Users/padmanabha/Projects/sensor_ai/results/models")
        self.results_path = Path("/Users/padmanabha/Projects/sensor_ai/results")
        
        # Load data and label encoder once
        print("Loading and preprocessing data...")
        (self.X_train, self.y_train), (self.X_test, self.y_test), self.le = load_and_preprocess()
        
        # Create test dataset and loader
        test_dataset = GestureDataset(self.X_test, self.y_test)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"Test data shape: {self.X_test.shape}")
        print(f"Number of classes: {len(self.le.classes_)}")
        print(f"Classes: {list(self.le.classes_)}")
        
    def get_model_architecture(self, model_name):
        """Get the model architecture based on model name."""
        model_name_lower = model_name.lower()
        
        # Check for CNN-LSTM first (before LSTM to avoid false matches)
        if "cnn_lstm" in model_name_lower or "cnnlstm" in model_name_lower:
            return CNNLSTMModel(input_size=6, num_classes=len(self.le.classes_))
        
        # Check for LSTM
        elif "lstm" in model_name_lower and "cnn" not in model_name_lower:
            return LSTMGestureModel(input_size=6, num_classes=len(self.le.classes_))
        
        # Check for CNN versions
        elif "gesturereccnn_v3" in model_name_lower or "v3" in model_name_lower:
            return GestureRecCNN_V3(input_size=6, num_classes=len(self.le.classes_))
        elif "gesturereccnn_v2" in model_name_lower or "v2" in model_name_lower:
            return GestureRecCNN_V2(input_size=6, num_classes=len(self.le.classes_))
        elif "gesturereccnn_v1" in model_name_lower or "v1" in model_name_lower:
            return GestureRecCNN_V1(input_size=6, num_classes=len(self.le.classes_))
        
        # Fallback: try to detect from filename patterns
        elif "cnn" in model_name_lower:
            # Default to V3 if just "cnn" is mentioned
            return GestureRecCNN_V3(input_size=6, num_classes=len(self.le.classes_))
        
        else:
            # Print available model files for debugging
            print(f"Could not determine architecture for: {model_name}")
            print("Available architectures: GestureRecCNN_V1, GestureRecCNN_V2, GestureRecCNN_V3, LSTMGestureModel, CNNLSTMModel")
            raise ValueError(f"Unknown model architecture: {model_name}")
    
    def load_model(self, model_path):
        """Load a model from file path."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Extract model name from file path
        model_filename = Path(model_path).stem
        
        # Remove common suffixes to get model architecture name
        model_name = model_filename.replace("_final_model", "").replace("_model", "").replace("best_", "")
        
        # Debug print
        print(f"Original filename: {model_filename}")
        print(f"Extracted model name: {model_name}")
        
        # Get model architecture
        model = self.get_model_architecture(model_name)
        
        # Load weights
        print(f"Loading model: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model, model_name
    
    def evaluate_single_model(self, model, model_name):
        """Evaluate a single model and print detailed results."""
        print(f"\nEvaluating {model_name}...")
        print("=" * 60)
        
        all_predictions = []
        all_actuals = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device).float()
                y_batch = y_batch.to(self.device)
                
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_actuals.extend(y_batch.cpu().numpy())
        
        # Calculate accuracy
        accuracy = 100 * correct / total
        print(f"Overall Accuracy: {accuracy:.2f}%")
        
        # Convert back to string labels
        pred_labels = self.le.inverse_transform(all_predictions)
        actual_labels = self.le.inverse_transform(all_actuals)
        
        # Print detailed evaluation (your requested format)
        print("\nDetailed Evaluation:")
        print("=" * 50)
        report = classification_report(actual_labels, pred_labels, digits=2)
        print(report)
        print("Confusion Matrix:")
        cm = confusion_matrix(actual_labels, pred_labels)
        print(cm)
        
        return accuracy, pred_labels, actual_labels, report, cm
    
    def evaluate_final_models_only(self):
        """Evaluate only the final models (not best models)."""
        if not self.models_path.exists():
            print(f"Models directory not found: {self.models_path}")
            return
        
        # Find only final model files
        final_model_files = list(self.models_path.glob("*_final_model.pth"))
        
        if not final_model_files:
            print(f"No final model files found in: {self.models_path}")
            print("Looking for files matching pattern: *_final_model.pth")
            return
        
        print(f"Found {len(final_model_files)} final model files:")
        for f in final_model_files:
            print(f"  - {f.name}")
        
        results = {}
        failed_models = []
        
        for model_file in final_model_files:
            try:
                print(f"\n{'='*60}")
                print(f"Processing: {model_file.name}")
                print('='*60)
                
                model, model_name = self.load_model(model_file)
                accuracy, pred_labels, actual_labels, report, cm = self.evaluate_single_model(model, model_name)
                
                results[model_file.name] = {
                    'model_name': model_name,
                    'accuracy': accuracy,
                    'model_path': str(model_file),
                    'classification_report': report,
                    'confusion_matrix': cm
                }
                
                print(f"✓ Successfully evaluated {model_file.name}")
                
            except Exception as e:
                print(f"✗ Error evaluating {model_file.name}: {e}")
                failed_models.append((model_file.name, str(e)))
                continue
        
        # Print summary of failed models
        if failed_models:
            print(f"\n{'='*60}")
            print("FAILED MODELS SUMMARY:")
            print('='*60)
            for model_name, error in failed_models:
                print(f"❌ {model_name}")
                print(f"   Error: {error[:100]}{'...' if len(error) > 100 else ''}")
                print()
        
        # Print summary comparison
        if results:
            print("\n" + "="*80)
            print("FINAL MODELS COMPARISON:")
            print("="*80)
            self.print_model_comparison(results)
        else:
            print("No final models were successfully evaluated!")
        
        # Save results to file
        self.save_results_to_file(results, failed_models, "final_models_evaluation.txt")
        
        return results
    
    def print_model_comparison(self, results):
        """Print comparison of all evaluated models."""
        if not results:
            return
        
        print(f"{'Rank':<5} {'Model File':<40} {'Architecture':<20} {'Accuracy':<10}")
        print("-" * 80)
        
        # Sort by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for rank, (filename, result) in enumerate(sorted_results, 1):
            print(f"{rank:<5} {filename:<40} {result['model_name']:<20} {result['accuracy']:<10.2f}%")
        
        print("\nBest performing model:")
        best_file, best_result = sorted_results[0]
        print(f"  File: {best_file}")
        print(f"  Architecture: {best_result['model_name']}")
        print(f"  Accuracy: {best_result['accuracy']:.2f}%")
        print(f"  Path: {best_result['model_path']}")

    def save_results_to_file(self, results, failed_models, output_filename="final_models_evaluation.txt"):
        """Save evaluation results to a text file in the results directory."""
        output_path = self.results_path / output_filename
        
        with open(output_path, 'w') as f:
            f.write("GESTURE RECOGNITION MODEL EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            # Write evaluation timestamp
            from datetime import datetime
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Data Shape: {self.X_test.shape}\n")
            f.write(f"Number of Classes: {len(self.le.classes_)}\n")
            f.write(f"Classes: {list(self.le.classes_)}\n\n")
            
            # Write detailed evaluations for each model
            if results:
                f.write("DETAILED MODEL EVALUATIONS:\n")
                f.write("=" * 60 + "\n\n")
                
                # Sort by accuracy for better presentation
                sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
                
                for filename, result in sorted_results:
                    f.write(f"Model: {filename}\n")
                    f.write(f"Architecture: {result['model_name']}\n")
                    f.write(f"Overall Accuracy: {result['accuracy']:.2f}%\n")
                    f.write(f"Path: {result['model_path']}\n\n")
                    
                    # Write detailed evaluation
                    f.write("Detailed Evaluation:\n")
                    f.write("=" * 50 + "\n")
                    f.write(result['classification_report'])
                    f.write("\nConfusion Matrix:\n")
                    
                    # Format confusion matrix nicely
                    cm = result['confusion_matrix']
                    f.write(str(cm) + "\n")
                    f.write("\n" + "=" * 60 + "\n\n")
                
                # Write comparison summary
                f.write("MODEL COMPARISON SUMMARY:\n")
                f.write("=" * 60 + "\n")
                f.write(f"{'Rank':<5} {'Model File':<35} {'Architecture':<20} {'Accuracy':<10}\n")
                f.write("-" * 75 + "\n")
                
                for rank, (filename, result) in enumerate(sorted_results, 1):
                    f.write(f"{rank:<5} {filename:<35} {result['model_name']:<20} {result['accuracy']:<10.2f}%\n")
                
                # Write best model info
                best_file, best_result = sorted_results[0]
                f.write(f"\nBest Performing Model:\n")
                f.write(f"  File: {best_file}\n")
                f.write(f"  Architecture: {best_result['model_name']}\n")
                f.write(f"  Accuracy: {best_result['accuracy']:.2f}%\n")
                f.write(f"  Path: {best_result['model_path']}\n\n")
            
            # Write failed models
            if failed_models:
                f.write("FAILED EVALUATIONS:\n")
                f.write("-" * 40 + "\n")
                for model_name, error in failed_models:
                    f.write(f"❌ {model_name}\n")
                    f.write(f"   Error: {error}\n\n")
            
            f.write(f"\nEvaluation Summary:\n")
            f.write(f"Total Models Evaluated: {len(results)}\n")
            f.write(f"Total Models Failed: {len(failed_models)}\n")
        
        print(f"\n📄 Results saved to: {output_path}")
        return output_path

def main():
    # Initialize evaluator
    evaluator = GestureModelEvaluator()
    
    # Evaluate final models only
    print("Evaluating final models...")
    results = evaluator.evaluate_final_models_only()

if __name__ == "__main__":
    main()