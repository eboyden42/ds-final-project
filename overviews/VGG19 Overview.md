VGG19 is different than InsightFace in the sense that it is only a single network, and any pre-processing will need to be completed using a separate library (in our case TorchVision does most of this).

## Preprocessing
The preprocessing done by TorchVision is similar but slightly less robust than that of InsightFace. It first compresses the image down using Bilinear Interpolation (a method that can be sort of thought like as averaging the pixels around each pixel sort of like a convolution). Then it resizes the image to $224 \times 224$ pixels which is exactly what is needed for the VGG19 model. The numbers are then rescaled, and normalized using specific means and standard deviations that are derived from the training data.
## Convolutional and max pooling layers
VGG19 starts with 10 layers, alternating between $3 \times 3$ convolutions and max pooling. The depth of the convolutions increase as the layers progress. The number of filters for each convolution increases by powers of $2$ starting from $64$ filters and capping at $512$ filters. Each convolutional layer is also paired with a ReLU activation function.

The max-pooling layers remain the same across the model, each having a $2x2$ filter size and a stride of $2$.
## Fully connected layers
The model is finished off with three massive neural network layers, the first two using $4096$ neurons and ReLU activations, and the final layer using $1000$ neurons with a softmax activation. For our purposes since we only want the 4096 vector embedding we'll be chopping off that last layer.