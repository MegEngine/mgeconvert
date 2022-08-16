import torch
import torchvision

model = torchvision.models.efficientnet_b0()
model.eval()
x = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, x, "efficientnet_b0.onnx", opset_version=12)
