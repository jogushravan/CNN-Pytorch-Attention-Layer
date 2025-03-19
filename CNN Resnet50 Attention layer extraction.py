import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
target_layer = model.backbone.body.layer4[-1]
print("NumPy is working:", np.__version__)
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

    def generate(self, image_tensor, bbox):
        output = self.model(image_tensor)[0]
        loss = output["scores"].sum()
        self.model.zero_grad()
        loss.backward()
        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (self.activations * gradients).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().detach().numpy()
        cam = cv2.resize(cam, (image_tensor.shape[2], image_tensor.shape[3]))
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

def preprocess_image(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))
        image = image.astype(np.uint8)  # Ensure NumPy conversion
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image).unsqueeze(0)
        return image, image_tensor

image_path = "dog1.jpg"
original_image, image_tensor = preprocess_image(image_path)
grad_cam = GradCAM(model, target_layer)
output = model(image_tensor)[0]
bboxes = output["boxes"].cpu().detach().numpy()
for bbox in bboxes:
    x1, y1, x2, y2 = bbox
    heatmap = grad_cam.generate(image_tensor, bbox)
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
cv2.imshow("Grad-CAM Faster R-CNN", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
