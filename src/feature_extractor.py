import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetExtractor(nn.Module):
    """ResNet-18 feature extractor (frozen backbone)."""
    
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(m.children())[:-1]).to(device)
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 128)),
            T.ToTensor(),
        ])
        
        # Freeze backbone parameters
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad = False
    
    def forward(self, img):
        """Extract 512-dim feature vector from image crop."""
        x = self.transform(img).unsqueeze(0).to(device)
        f = self.features(x)
        return f.view(-1)