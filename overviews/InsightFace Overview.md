The insight face model is spit into a few steps that eventually produce a `512 d` vector embedding, given an image.
## Detection
In this stage InsightFace uses a facial detection model that scans an image and produces a set of coordinates, representing bounding boxes, along with a corresponding `det_score`. This score is a number, ranged $0$ to  $1$, that represents how confident the model is that there is a face present (higher scores meaning more certainty).  Then the bounding boxes are filtered out based on some threshold `det_score` set by the user.

In this stage there are also 5 key facial features that are identified via coordinates. These are the jawline, eyebrows, nose, eyes, and mouth. These features are extracted into a vector and are crucial for the next step.
## Normalization
During the normalization phase, the model uses locations of the bounding boxes and the facial features to create a standard $112\times 112$ image that always places the eyes and nose at the same coordinates. This makes the job of the embedding network much easier.
## Embedding
Previously, this embedding model has been trained using `ArcFace` (or one of its variations) which is a loss function that uses the angle between vectors as a distance metric, and normalizes the vectors to a length of $1$ (creating a hyper-sphere). The long and the short of it is that apparently this loss function forces the underlying network backbone to be more precise about it's vector embeddings, which can come in handy.

The underlying network backbones are one of "IResNet, MobilefaceNet, MobileNet, InceptionResNet_v2, DenseNet" and more.  When we use the python package we are using a model that has been pretrained (using `ArcFace`) to produce a `512 d` vector embedding.

A popular backbone model is IResNet (Improved Residual Network), that uses a combination of convolutions, batch normalizations, and ReLU activation functions. 
