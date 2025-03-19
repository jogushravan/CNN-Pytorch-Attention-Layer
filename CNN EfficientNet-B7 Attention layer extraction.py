import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
model.eval()
target_layer = model.features[-1]

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image_tensor, class_idx):
        output = self.model(image_tensor)
        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward()
        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (self.activations * gradients).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().detach().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return image, image_tensor

image_path = "your_image.jpg"
original_image, image_tensor = preprocess_image(image_path)
grad_cam = GradCAM(model, target_layer)
output = model(image_tensor)
class_idx = torch.argmax(output).item()
heatmap = grad_cam.generate(image_tensor, class_idx)
heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
overlay = cv2.addWeighted((original_image * 255).astype(np.uint8), 0.6, heatmap, 0.4, 0)
cv2.imshow("Grad-CAM EfficientNet-B7", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
