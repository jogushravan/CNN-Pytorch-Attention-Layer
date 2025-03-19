# CNN-Pytorch

Convolutional Neural Network (CNN) with an attention mechanism (Squeeze-and-Excitation block) to highlight important regions in an image. The generated attention heatmap visually shows which parts of the image the model focuses on.

###Attention Layer (Squeeze-and-Excitation Block)
filters = input_tensor.shape[-1]  # Get number of channels (features) 
se = layers.GlobalAveragePooling2D()(input_tensor)  # Step 1: "Squeeze" feature maps into a single vector
se = layers.Dense(filters // ratio, activation="relu")(se)  # Step 2: Reduce the size to focus on important features
se = layers.Dense(filters, activation="sigmoid")(se)  # Step 3: Scale features between 0 and 1
se = layers.Reshape((1, 1, filters))(se)  # Step 4: Reshape it to match the original feature map shape

return layers.multiply([input_tensor, se])  # Step 5: Multiply attention scores with input features
