import torch
import numpy as np
import onnxruntime as ort
import sys
from pathlib import Path

def patch_easyocr_for_onnx_export():
    try:
        import easyocr.model.modules as easyocr_modules
    except Exception:
        return
    if not hasattr(easyocr_modules, 'BidirectionalLSTM'):
        return
    def forward_no_flatten(self, input):
        recurrent, _ = self.rnn(input)
        output = self.linear(recurrent)
        return output
    easyocr_modules.BidirectionalLSTM.forward = forward_no_flatten

def patch_easyocr_adaptive_pooling(model):
    import torch.nn as nn
    class OnnxFriendlyAdaptivePool(nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.output_size = output_size
        def forward(self, x):
            return x.mean(dim=-1, keepdim=True)
    for name, module in model.named_modules():
        if isinstance(module, nn.AdaptiveAvgPool2d):
            if module.output_size == (None, 1):
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], OnnxFriendlyAdaptivePool(module.output_size))

def compare(width):
    import easyocr
    models_dir = Path("./models").resolve()
    reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=str(models_dir), download_enabled=False)
    patch_easyocr_for_onnx_export()
    recognizer = reader.recognizer if hasattr(reader, 'recognizer') else reader.recog_network
    patch_easyocr_adaptive_pooling(recognizer)
    recognizer.eval()

    onnx_path = models_dir / "onnx" / "english_g2.onnx"
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    input_shape = (1, 1, 32, width)
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)
    text_input = torch.zeros((1, 1), dtype=torch.long)

    with torch.no_grad():
        torch_output = recognizer(dummy_input, text_input)

    input_names = [si.name for si in session.get_inputs()]
    feeds = {input_names[0]: dummy_input.numpy()}
    if len(input_names) > 1:
        feeds[input_names[1]] = text_input.numpy()

    onnx_outputs = session.run(None, feeds)
    onnx_output = onnx_outputs[0]

    torch_np = torch_output.detach().cpu().numpy()
    diff = np.abs(torch_np - onnx_output)
    print(f"Width: {width}, Max diff: {diff.max():.6f}, Shapes: {torch_np.shape} vs {onnx_output.shape}")
    return np.allclose(torch_np, onnx_output, atol=1e-3, rtol=1e-3)

if __name__ == "__main__":
    for w in [400, 600, 800, 1000]:
        compare(w)
