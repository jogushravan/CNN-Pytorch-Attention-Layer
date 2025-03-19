import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import matplotlib.pyplot as plt

#pip install torch torchvision numpy matplotlib opencv-python
# 📌 Load the pretrained EfficientNetV2-L model
model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
model.eval()  # Set the model to evaluation mode

# 📌 Identify the last convolutional layer for Grad-CAM
# In EfficientNetV2, the last feature extraction layer is "features[-1]"
target_layer = model.features[-1]  # Last conv layer before classification


# 📌 Grad-CAM Class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to capture activations (feature maps)
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        """ Saves feature maps (activations) from the target layer. """
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        """ Saves gradients during backpropagation. """
        self.gradients = grad_output[0]

    def generate(self, image_tensor, class_idx):
        """ Computes Grad-CAM heatmap for a specific class index. """
        output = self.model(image_tensor)  # Forward pass
        loss = output[:, class_idx]  # Get prediction for the target class
        self.model.zero_grad()  # Reset gradients
        loss.backward()  # Compute gradients

        # Compute Grad-CAM heatmap
        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)  # Average across spatial dimensions
        cam = (self.activations * gradients).sum(dim=1, keepdim=True)  # Weighted sum
        cam = torch.relu(cam)  # Remove negative values
        cam = cam.squeeze().cpu().detach().numpy()  # Convert to numpy
        cam = cv2.resize(cam, (224, 224))  # Resize to match image size
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize

        return cam


# 📌 Function to Load and Preprocess Image
def preprocess_image(image_path):
    """ Load and preprocess an image for EfficientNetV2 """
    image = cv2.imread(image_path)  # Load image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (224, 224))  # Resize for model
    image = image.astype(np.float32) / 255.0  # Normalize pixel values
    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Convert to PyTorch tensor
    return image, image_tensor


# 📌 Load an image
image_path = "your_image.jpg"  # ⚠️ Replace with your local image path!
original_image, image_tensor = preprocess_image(image_path)

# 📌 Initialize Grad-CAM
grad_cam = GradCAM(model, target_layer)

# 📌 Run model forward to get class predictions
output = model(image_tensor)
class_idx = torch.argmax(output).item()  # Get most confident class index

# 📌 Generate Grad-CAM heatmap
heatmap = grad_cam.generate(image_tensor, class_idx)


# 📌 Function to Overlay Heatmap on Image
def overlay_heatmap(image, heatmap):
    """ Overlay heatmap on image """
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)  # Convert to color heatmap
    overlay = cv2.addWeighted((image * 255).astype(np.uint8), 0.6, heatmap, 0.4, 0)  # Blend with original image
    return overlay


# 📌 Generate overlay
overlayed_image = overlay_heatmap(original_image, heatmap)

# 📌 Display Results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(original_image)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(overlayed_image)
ax[1].set_title("Grad-CAM Attention Map")
ax[1].axis("off")

plt.show()
