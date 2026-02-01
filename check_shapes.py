import torch
import easyocr
from pathlib import Path

reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./models', download_enabled=False)
recognizer = reader.recognizer if hasattr(reader, 'recognizer') else reader.recog_network

class Hook:
    def __init__(self):
        self.input_shape = None
        self.output_shape = None
    def __call__(self, module, input, output):
        self.input_shape = input[0].shape
        self.output_shape = output.shape

hook = Hook()
recognizer.AdaptiveAvgPool.register_forward_hook(hook)

dummy_input = torch.randn(1, 1, 32, 100)
dummy_text = torch.zeros(1, 1).long()

with torch.no_grad():
    recognizer(dummy_input, dummy_text)

print(f"Pooling Input Shape: {hook.input_shape}")
print(f"Pooling Output Shape: {hook.output_shape}")
