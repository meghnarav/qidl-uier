import torch
from PIL import Image
from torchvision import transforms
from models import QIDL
from config import Config

device=Config.device

model=QIDL().to(device)
model.load_state_dict(torch.load("qidl.pth"))
model.eval()

t=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

img=Image.open("test.jpg").convert("RGB")
x=t(img).unsqueeze(0).to(device)

with torch.no_grad():
    y=model(x)

y=((y.squeeze().cpu()+1)/2).clamp(0,1)
Image.fromarray((y.numpy().transpose(1,2,0)*255).astype("uint8")).save("out.jpg")