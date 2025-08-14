import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Import your models with correct path
try:
    import sys
    sys.path.append('.')  # Add current directory to path
    from model_defs.model_defs import GestureRecCNN_V1, GestureRecCNN_V2, GestureRecCNN_V3, LSTMGestureModel, CNNLSTMModel
    MODELS_IMPORTED = True
    print("✅ Successfully imported all models")
except ImportError as e:
    print(f"❌ Could not import models: {e}")
    print("Continuing with basic visualization...")
    MODELS_IMPORTED = False

# Create results directory
RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

def create_horizontal_architecture_diagram(model, model_name, input_shape=(1, 200, 6), max_boxes_per_row=10):
    """
    Create a horizontal visualization of the model architecture with row wrapping
    """
    # Get model summary information
    layers_info = get_model_layers_info(model, model_name)
    
    # Calculate number of rows needed
    num_rows = (len(layers_info) + max_boxes_per_row - 1) // max_boxes_per_row
    
    # Set up the figure with appropriate size for multiple rows
    fig_height = 6 + (num_rows - 1) * 4  # Base height + extra for each additional row
    fig, ax = plt.subplots(1, 1, figsize=(18, fig_height))
    
    # Set plot limits
    ax.set_xlim(0, max_boxes_per_row + 1)
    ax.set_ylim(0, num_rows * 4 + 2)
    ax.axis('off')
    
    # Colors for different layer types
    colors = {
        'input': '#e1f5fe',
        'conv': '#fff3e0', 
        'pool': '#ffebee',
        'activation': '#f1f8e9',
        'linear': '#f3e5f5',
        'lstm': '#e8f5e8',
        'output': '#c8e6c9',
        'reshape': '#fce4ec'
    }
    
    # Draw layers with wrapping
    for i, layer_info in enumerate(layers_info):
        # Calculate row and column position
        row = i // max_boxes_per_row
        col = i % max_boxes_per_row
        
        # Calculate positions (start from top)
        x_pos = col + 1
        y_pos = (num_rows - row - 1) * 4 + 2  # Flip y-coordinate to start from top
        
        # Determine box properties
        layer_type = layer_info['type']
        color = colors.get(layer_type, '#f5f5f5')
        
        # Create fancy box
        box = FancyBboxPatch(
            (x_pos - 0.4, y_pos - 0.8), 0.8, 1.6,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(box)
        
        # Add layer text
        ax.text(x_pos, y_pos + 0.4, layer_info['name'], 
                ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(x_pos, y_pos, layer_info['details'], 
                ha='center', va='center', fontsize=7)
        ax.text(x_pos, y_pos - 0.4, layer_info['shape'], 
                ha='center', va='center', fontsize=6, style='italic')
        
        # Draw arrows between layers
        if i < len(layers_info) - 1:
            next_row = (i + 1) // max_boxes_per_row
            next_col = (i + 1) % max_boxes_per_row
            
            if row == next_row:  # Same row - horizontal arrow
                next_x_pos = next_col + 1
                arrow = patches.FancyArrowPatch(
                    (x_pos + 0.4, y_pos), 
                    (next_x_pos - 0.4, y_pos),
                    arrowstyle='->', 
                    mutation_scale=15, 
                    color='black'
                )
                ax.add_patch(arrow)
            else:  # Different row - curved arrow going down
                next_x_pos = next_col + 1
                next_y_pos = (num_rows - next_row - 1) * 4 + 2
                
                # Create a curved arrow path
                arrow = patches.FancyArrowPatch(
                    (x_pos, y_pos - 0.8), 
                    (next_x_pos, next_y_pos + 0.8),
                    arrowstyle='->', 
                    mutation_scale=15, 
                    color='red',
                    connectionstyle="arc3,rad=0.3"
                )
                ax.add_patch(arrow)
    
    # Add title at the top
    title_y = num_rows * 4 + 1
    ax.text((max_boxes_per_row + 1) / 2, title_y, f'{model_name} Architecture', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add input/output labels if there are layers
    if layers_info:
        # Input label for first box
        first_row = (len(layers_info) - 1) // max_boxes_per_row
        first_y = (num_rows - first_row - 1) * 4 + 2
        ax.text(1, first_y - 1.2, 'INPUT', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='blue')
        
        # Output label for last box
        last_idx = len(layers_info) - 1
        last_row = last_idx // max_boxes_per_row
        last_col = last_idx % max_boxes_per_row
        last_x = last_col + 1
        last_y = (num_rows - last_row - 1) * 4 + 2
        ax.text(last_x, last_y - 1.2, 'OUTPUT', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='green')
    
    plt.tight_layout()
    
    # Save the diagram
    output_path = RESULTS / f"{model_name.lower().replace(' ', '_')}_architecture.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Architecture diagram saved to: {output_path}")
    return output_path

def get_model_layers_info(model, model_name):
    """
    Extract layer information from the model for visualization
    """
    layers_info = []
    
    print(f"🔍 Matching model name: '{model_name}'")  # Debug print
    
    # Convert to lowercase for easier matching
    model_name_lower = model_name.lower()
    
    if "gesturereccnn_v1" in model_name_lower or "gesturereccnn v1" in model_name_lower:
        layers_info = [
            {'name': 'Input', 'type': 'input', 'details': '(batch, 200, 6)', 'shape': 'Time Series'},
            {'name': 'Permute', 'type': 'reshape', 'details': '→ (batch, 6, 200)', 'shape': 'Reshape'},
            {'name': 'Conv1D', 'type': 'conv', 'details': '6→32\nk=3, p=1', 'shape': '(batch, 32, 200)'},
            {'name': 'ReLU', 'type': 'activation', 'details': 'Activation', 'shape': '(batch, 32, 200)'},
            {'name': 'MaxPool', 'type': 'pool', 'details': 'k=2', 'shape': '(batch, 32, 100)'},
            {'name': 'Conv1D', 'type': 'conv', 'details': '32→64\nk=3, p=1', 'shape': '(batch, 64, 100)'},
            {'name': 'ReLU', 'type': 'activation', 'details': 'Activation', 'shape': '(batch, 64, 100)'},
            {'name': 'MaxPool', 'type': 'pool', 'details': 'k=2', 'shape': '(batch, 64, 50)'},
            {'name': 'Conv1D', 'type': 'conv', 'details': '64→128\nk=3, p=1', 'shape': '(batch, 128, 50)'},
            {'name': 'ReLU', 'type': 'activation', 'details': 'Activation', 'shape': '(batch, 128, 50)'},
            {'name': 'AdaptiveAvg\nPool1D', 'type': 'pool', 'details': '→ size=1', 'shape': '(batch, 128, 1)'},
            {'name': 'Squeeze', 'type': 'reshape', 'details': 'Remove dim', 'shape': '(batch, 128)'},
            {'name': 'Linear', 'type': 'linear', 'details': '128→64', 'shape': '(batch, 64)'},
            {'name': 'ReLU', 'type': 'activation', 'details': 'Activation', 'shape': '(batch, 64)'},
            {'name': 'Linear', 'type': 'linear', 'details': '64→classes', 'shape': '(batch, n_classes)'},
            {'name': 'Output', 'type': 'output', 'details': 'Predictions', 'shape': 'Classification'}
        ]
    
    elif "gesturereccnn_v2" in model_name_lower or "gesturereccnn v2" in model_name_lower:
        layers_info = [
            {'name': 'Input', 'type': 'input', 'details': '(batch, 200, 6)', 'shape': 'Time Series'},
            {'name': 'Permute', 'type': 'reshape', 'details': '→ (batch, 6, 200)', 'shape': 'Reshape'},
            {'name': 'Conv1D', 'type': 'conv', 'details': '6→64\nk=5, p=2', 'shape': '(batch, 64, 200)'},
            {'name': 'BatchNorm', 'type': 'activation', 'details': 'Normalize', 'shape': '(batch, 64, 200)'},
            {'name': 'ReLU', 'type': 'activation', 'details': 'Activation', 'shape': '(batch, 64, 200)'},
            {'name': 'MaxPool', 'type': 'pool', 'details': 'k=2', 'shape': '(batch, 64, 100)'},
            {'name': 'Conv1D', 'type': 'conv', 'details': '64→128\nk=3, p=1', 'shape': '(batch, 128, 100)'},
            {'name': 'BatchNorm', 'type': 'activation', 'details': 'Normalize', 'shape': '(batch, 128, 100)'},
            {'name': 'ReLU', 'type': 'activation', 'details': 'Activation', 'shape': '(batch, 128, 100)'},
            {'name': 'MaxPool', 'type': 'pool', 'details': 'k=2', 'shape': '(batch, 128, 50)'},
            {'name': 'AdaptiveAvg\nPool1D', 'type': 'pool', 'details': '→ size=1', 'shape': '(batch, 128, 1)'},
            {'name': 'Dropout', 'type': 'activation', 'details': 'p=0.5', 'shape': '(batch, 128)'},
            {'name': 'Linear', 'type': 'linear', 'details': '128→64', 'shape': '(batch, 64)'},
            {'name': 'ReLU', 'type': 'activation', 'details': 'Activation', 'shape': '(batch, 64)'},
            {'name': 'Linear', 'type': 'linear', 'details': '64→classes', 'shape': '(batch, n_classes)'},
            {'name': 'Output', 'type': 'output', 'details': 'Predictions', 'shape': 'Classification'}
        ]
    
    elif "gesturereccnn_v3" in model_name_lower or "gesturereccnn v3" in model_name_lower:
        layers_info = [
            {'name': 'Input', 'type': 'input', 'details': '(batch, 200, 6)', 'shape': 'Time Series'},
            {'name': 'Permute', 'type': 'reshape', 'details': '→ (batch, 6, 200)', 'shape': 'Reshape'},
            {'name': 'Conv1D', 'type': 'conv', 'details': '6→64\nk=7, p=3', 'shape': '(batch, 64, 200)'},
            {'name': 'BatchNorm', 'type': 'activation', 'details': 'Normalize', 'shape': '(batch, 64, 200)'},
            {'name': 'ReLU', 'type': 'activation', 'details': 'Activation', 'shape': '(batch, 64, 200)'},
            {'name': 'MaxPool', 'type': 'pool', 'details': 'k=3, s=2', 'shape': '(batch, 64, 99)'},
            {'name': 'Conv1D', 'type': 'conv', 'details': '64→128\nk=5, p=2', 'shape': '(batch, 128, 99)'},
            {'name': 'BatchNorm', 'type': 'activation', 'details': 'Normalize', 'shape': '(batch, 128, 99)'},
            {'name': 'ReLU', 'type': 'activation', 'details': 'Activation', 'shape': '(batch, 128, 99)'},
            {'name': 'MaxPool', 'type': 'pool', 'details': 'k=3, s=2', 'shape': '(batch, 128, 49)'},
            {'name': 'Conv1D', 'type': 'conv', 'details': '128→256\nk=3, p=1', 'shape': '(batch, 256, 49)'},
            {'name': 'BatchNorm', 'type': 'activation', 'details': 'Normalize', 'shape': '(batch, 256, 49)'},
            {'name': 'ReLU', 'type': 'activation', 'details': 'Activation', 'shape': '(batch, 256, 49)'},
            {'name': 'AdaptiveAvg\nPool1D', 'type': 'pool', 'details': '→ size=1', 'shape': '(batch, 256, 1)'},
            {'name': 'Dropout', 'type': 'activation', 'details': 'p=0.5', 'shape': '(batch, 256)'},
            {'name': 'Linear', 'type': 'linear', 'details': '256→128', 'shape': '(batch, 128)'},
            {'name': 'ReLU', 'type': 'activation', 'details': 'Activation', 'shape': '(batch, 128)'},
            {'name': 'Dropout', 'type': 'activation', 'details': 'p=0.3', 'shape': '(batch, 128)'},
            {'name': 'Linear', 'type': 'linear', 'details': '128→classes', 'shape': '(batch, n_classes)'},
            {'name': 'Output', 'type': 'output', 'details': 'Predictions', 'shape': 'Classification'}
        ]
    
    elif "cnn" in model_name_lower and "lstm" in model_name_lower:
        layers_info = [
            {'name': 'Input', 'type': 'input', 'details': '(batch, 200, 6)', 'shape': 'Time Series'},
            {'name': 'Permute', 'type': 'reshape', 'details': '→ (batch, 6, 200)', 'shape': 'Reshape'},
            {'name': 'Conv1D', 'type': 'conv', 'details': '6→32\nk=3, p=1', 'shape': '(batch, 32, 200)'},
            {'name': 'ReLU', 'type': 'activation', 'details': 'Activation', 'shape': '(batch, 32, 200)'},
            {'name': 'MaxPool', 'type': 'pool', 'details': 'k=2', 'shape': '(batch, 32, 100)'},
            {'name': 'Conv1D', 'type': 'conv', 'details': '32→64\nk=3, p=1', 'shape': '(batch, 64, 100)'},
            {'name': 'ReLU', 'type': 'activation', 'details': 'Activation', 'shape': '(batch, 64, 100)'},
            {'name': 'MaxPool', 'type': 'pool', 'details': 'k=2', 'shape': '(batch, 64, 50)'},
            {'name': 'Permute', 'type': 'reshape', 'details': '→ (batch, 50, 64)', 'shape': 'Reshape'},
            {'name': 'LSTM', 'type': 'lstm', 'details': '64→128\nhidden', 'shape': '(batch, 50, 128)'},
            {'name': 'Last Hidden', 'type': 'reshape', 'details': 'h_n[-1]', 'shape': '(batch, 128)'},
            {'name': 'Linear', 'type': 'linear', 'details': '128→64', 'shape': '(batch, 64)'},
            {'name': 'ReLU', 'type': 'activation', 'details': 'Activation', 'shape': '(batch, 64)'},
            {'name': 'Linear', 'type': 'linear', 'details': '64→classes', 'shape': '(batch, n_classes)'},
            {'name': 'Output', 'type': 'output', 'details': 'Predictions', 'shape': 'Classification'}
        ]
    
    elif "lstm" in model_name_lower and "cnn" not in model_name_lower:
        layers_info = [
            {'name': 'Input', 'type': 'input', 'details': '(batch, 200, 6)', 'shape': 'Time Series'},
            {'name': 'LSTM', 'type': 'lstm', 'details': '6→128\nhidden', 'shape': '(batch, 200, 128)'},
            {'name': 'Last Hidden', 'type': 'reshape', 'details': 'h_n[-1]', 'shape': '(batch, 128)'},
            {'name': 'Linear', 'type': 'linear', 'details': '128→64', 'shape': '(batch, 64)'},
            {'name': 'ReLU', 'type': 'activation', 'details': 'Activation', 'shape': '(batch, 64)'},
            {'name': 'Linear', 'type': 'linear', 'details': '64→classes', 'shape': '(batch, n_classes)'},
            {'name': 'Output', 'type': 'output', 'details': 'Predictions', 'shape': 'Classification'}
        ]
    
    else:
        # Generic layer info for unknown models
        print(f"⚠️  No specific architecture found for: {model_name}")
        print(f"    Available patterns: gesturereccnn_v1, gesturereccnn_v2, gesturereccnn_v3, cnn+lstm, lstm")
        layers_info = [
            {'name': 'Input', 'type': 'input', 'details': 'Input Layer', 'shape': '(batch, ...)'},
            {'name': 'Hidden', 'type': 'conv', 'details': 'Processing\nLayers', 'shape': '(batch, ...)'},
            {'name': 'Output', 'type': 'output', 'details': 'Output Layer', 'shape': '(batch, classes)'}
        ]
    
    print(f"✅ Selected architecture with {len(layers_info)} layers")
    return layers_info

def visualize_saved_models():
    """
    Load saved models and create architecture diagrams
    """
    models_dir = Path("results/models")
    
    # Define model constructors - updated to match your actual filenames
    if MODELS_IMPORTED:
        model_constructors = {
            'gesturereccnn_v1': lambda: GestureRecCNN_V1(input_size=6, num_classes=2),
            'gesturereccnn_v2': lambda: GestureRecCNN_V2(input_size=6, num_classes=2), 
            'gesturereccnn_v3': lambda: GestureRecCNN_V3(input_size=6, num_classes=2),
            'lstm': lambda: LSTMGestureModel(input_size=6, num_classes=2),
            'cnn_lstm': lambda: CNNLSTMModel(input_size=6, num_classes=2)
        }
    else:
        model_constructors = {}
    
    if not models_dir.exists():
        print(f"Models directory {models_dir} not found.")
        print("Creating sample diagrams instead...")
        
        # Create sample diagrams for known models
        sample_models = ['GestureRecCNN_V1', 'GestureRecCNN_V2', 'GestureRecCNN_V3', 'CNN-LSTM', 'LSTM']
        for model_name in sample_models:
            create_horizontal_architecture_diagram(None, model_name)
        return
    
    # Look for saved model files
    model_files = list(models_dir.glob("*_final_model.pth"))
    print(f"Found model files: {[f.name for f in model_files]}")
    
    for model_file in model_files:
        # Extract model name from filename
        model_key = model_file.stem.replace('_final_model', '')
        print(f"Processing: {model_key}")
        
        if model_key in model_constructors and MODELS_IMPORTED:
            try:
                # Create model and load weights
                model = model_constructors[model_key]()
                model.load_state_dict(torch.load(model_file, map_location='cpu'))
                model.eval()
                
                # Create diagram
                model_display_name = model_key.replace('_', ' ').title().replace('Cnn', 'CNN').replace('Lstm', 'LSTM')
                create_horizontal_architecture_diagram(model, model_display_name)
                print(f"✅ Created diagram for {model_display_name}")
                
            except Exception as e:
                print(f"❌ Error processing {model_file}: {e}")
                # Create diagram without loading weights
                model_display_name = model_key.replace('_', ' ').title().replace('Cnn', 'CNN').replace('Lstm', 'LSTM')
                create_horizontal_architecture_diagram(None, model_display_name)
                print(f"✅ Created fallback diagram for {model_display_name}")
        else:
            print(f"⚠️  Creating diagram for {model_key} (model not imported)")
            # Create diagram based on filename pattern
            if 'gesturereccnn_v1' in model_key:
                model_display_name = 'GestureRecCNN V1'
            elif 'gesturereccnn_v2' in model_key:
                model_display_name = 'GestureRecCNN V2'  
            elif 'gesturereccnn_v3' in model_key:
                model_display_name = 'GestureRecCNN V3'
            elif 'cnn_lstm' in model_key:
                model_display_name = 'CNN-LSTM'
            elif 'lstm' in model_key:
                model_display_name = 'LSTM'
            else:
                model_display_name = model_key.replace('_', ' ').title()
                
            create_horizontal_architecture_diagram(None, model_display_name)
            print(f"✅ Created diagram for {model_display_name}")

def main():
    print("Creating model architecture visualizations...")
    print(f"Saving diagrams to: {RESULTS}")
    
    # Try to visualize saved models
    visualize_saved_models()
    
    print("\nArchitecture diagrams created successfully!")
    print(f"Check the {RESULTS} directory for PNG files.")

if __name__ == "__main__":
    main()