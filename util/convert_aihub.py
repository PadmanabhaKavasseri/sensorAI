import torch
# from enhanced_model import EnhancedGestureCNN
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # adds sensor_ai root

from model_defs.model_defs import EnhancedGestureCNN
import config
  # Replace with your model file
import qai_hub as hub

model_path = config.MODELS / "enhanced_cnn_gesture_model.pth"

# Load trained PyTorch model
model = EnhancedGestureCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Trace or script the model with example input
example_input = torch.randn(1, 200, 6)  # Update based on your actual input
pt_model = torch.jit.trace(model, example_input)



compile_job = hub.submit_compile_job(
    pt_model,
    name="GestureClassifier",
    device=hub.Device("RB3 Gen 2 (Proxy)"),  # Or another target
    input_specs=dict(input=example_input.shape),       # Match your model input name/shape
)

assert isinstance(compile_job, hub.CompileJob)
