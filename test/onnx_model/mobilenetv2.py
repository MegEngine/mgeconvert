import torch
import torchvision

model = torchvision.models.mobilenet_v2()
model.eval()
x = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, x, "mobilenetv2.onnx", opset_version=12)
