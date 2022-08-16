import torch
import torchvision

model = torchvision.models.mobilenet_v3_large()
model.eval()
x = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, x, "mobilenetv3.onnx", opset_version=12)
