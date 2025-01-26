import hydra
from omegaconf import DictConfig
import torch
import torchvision.transforms as transforms
from PIL import Image
import logging

log = logging.getLogger(__name__)



# Image loader 

def load_image(image_path: str, image_size: int, mean: list, std: list):
    """Load and preprocess the image"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


# Load the model 

def load_model(cfg: DictConfig):
    """Load the model from PyTorch Hub"""
    model = torch.hub.load('pytorch/vision:v0.10.0', 
                          cfg.model.name, 
                          pretrained=cfg.model.pretrained)
    model.eval()
    return model


# Prediction and Classify 

def get_prediction(model, image_tensor):
    """Get prediction for the image"""
    with torch.no_grad():
        output = model(image_tensor)
        
    # Load ImageNet class labels
    labels_path = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    labels = torch.hub.download_url_to_file(labels_path, 'imagenet_classes.txt', progress=False)
    
    with open('imagenet_classes.txt', 'r') as f:
        categories = [s.strip() for s in f.readlines()]
    
    # Get top predictions
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    return [(categories[top5_catid[i]], float(top5_prob[i])) for i in range(5)]

# config path for the configurations 
# log them also 

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function"""
    log.info(f"Configuration:\n{cfg}")
    
    # Load and preprocess image
    image_tensor = load_image(
        cfg.image.path,
        cfg.image.size,
        cfg.preprocessing.mean,
        cfg.preprocessing.std
    )
    
    # Load model
    model = load_model(cfg)
    
    # Get predictions
    predictions = get_prediction(model, image_tensor)
    
    # Print results
    log.info("\nTop 5 predictions:")
    for class_name, prob in predictions:
        log.info(f"{class_name}: {prob:.4f}")

if __name__ == "__main__":
    main() 