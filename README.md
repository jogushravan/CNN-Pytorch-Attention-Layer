Key Goals
  #. Quantitative, Qualitative Articulate --> Better Explainability
  
  #. Merchandise--> Optimize package design -->  boost sales [Optimize Merchandise strategy] and e-commerce visuals[digital AI platform]
  
  #. What consumer driving factor of sales
  
  --> Boost sales across channels
  
  #. PPS raking score is a key metric holistic view of overall pack performance
  
  #. Compare against competitors

### **Grad-CAM: Gradient-weighted Class Activation Mapping**
![image](https://github.com/user-attachments/assets/c1d00dec-1f9e-4b3b-8bd6-d6b80801447f)

1️⃣ Use YOLO to detect products on the shelf.

2️⃣ Use EfficientNet-B7 to extract features from each detected product.

3️⃣ Use Grad-CAM to generate attention maps for key regions.

4️⃣ Combine CNN embeddings with text embeddings for multimodal KPI prediction.

5️⃣ Send embeddings to an LLM (GPT-4) for detailed insights.

![image](https://github.com/user-attachments/assets/0502dd28-05fc-4b1e-b81d-817dc45efa13)


# CNN-Pytorch
Convolutional Neural Network (CNN) with an attention mechanism (Squeeze-and-Excitation block) to highlight important regions in an image. 
The generated attention heatmap visually shows which parts of the image the model focuses on.

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

### Multimodal Learning Pipeline CCN+LLM Embeddings with Fusion Layer.ipynb
Business Goal

1️⃣ Extract visual features from product images using CNN (EfficientNet-B7).

2️⃣ Extract text embeddings from product descriptions using LLM (e.g., OpenAI GPT, Mistral, or Llama).

3️⃣ Combine both embeddings for rich, context-aware predictions (e.g., shelf visibility, consumer interaction).

4️⃣ Send embeddings to an LLM API to generate detailed insights about a product’s performance.
End-to-End Multimodal Learning Pipeline with Trainable Fusion Layer
We will use a trainable fusion model to combine CNN-based image embeddings (EfficientNet-B7) and LLM-based text embeddings (OpenAI GPT, Mistral, or Llama). This will allow context-aware KPI predictions such as shelf visibility, consumer interaction, and buying patterns.

