# CNN-Pytorch

Convolutional Neural Network (CNN) with an attention mechanism (Squeeze-and-Excitation block) to highlight important regions in an image. The generated attention heatmap visually shows which parts of the image the model focuses on.

###Attention Layer (Squeeze-and-Excitation Block)

filters = input_tensor.shape[-1]  # Get number of channels (features) 

se = layers.GlobalAveragePooling2D()(input_tensor)  # Step 1: "Squeeze" feature maps into a single vector

se = layers.Dense(filters // ratio, activation="relu")(se)  # Step 2: Reduce the size to focus on important features

se = layers.Dense(filters, activation="sigmoid")(se)  # Step 3: Scale features between 0 and 1

se = layers.Reshape((1, 1, filters))(se)  # Step 4: Reshape it to match the original feature map shape

return layers.multiply([input_tensor, se])  # Step 5: Multiply attention scores with input features

### Full Grad-CAM Implementation for EfficientNetV2-L in PyTorch.py
✅ Loads EfficientNetV2-L with ImageNet weights.

✅ Extracts feature maps from the last convolutional layer.

✅ Computes gradients to find the most influential regions.

✅ Generates a heatmap that highlights the "important" parts of the image.

✅ Overlays the attention map on the original image for visualization.

### CNN EfficientNet-B7 Attention layer extraction.py
✅ Loads EfficientNet_B7.

✅ Extracts attention layers using Grad-CAM.

✅ Creates a heatmap overlay.

✅ Outputs regions the CNN focused on.
