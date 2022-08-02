import torch
import torchvision

model = torchvision.models.resnet18()
model.eval()
x = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, x, "resnet18.onnx", opset_version=12)
