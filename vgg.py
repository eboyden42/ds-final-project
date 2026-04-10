import torch
import torchvision.models as models
from PIL import Image
import os

# 1. Setup Model and Official Transforms
weights = models.VGG19_Weights.IMAGENET1K_V1
model = models.vgg19(weights=weights)
model.classifier = model.classifier[:-1]
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

preprocess = weights.transforms()

image_dir = './data/gallery/test' 
embeddings_dict = {}
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

for filename in os.listdir(image_dir):
    img_path = os.path.join(image_dir, filename)
        
    try:
        img = Image.open(img_path).convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0).to(device)
            
        with torch.no_grad():
            vector = model(input_tensor)
            
        embeddings_dict[filename] = vector.squeeze().cpu().numpy()
        print(f"processed {filename}")
            
    except Exception as e:
        print(f"Could not process {filename}: {e}")

print(f"processed {len(embeddings_dict)} images")

first_image = list(embeddings_dict.keys())[0]
print(f"vector for {first_image}: {embeddings_dict[first_image]}...")
print(f"vector length: {len(embeddings_dict[first_image])}")