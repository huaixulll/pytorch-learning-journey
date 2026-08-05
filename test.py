import torch
import torchvision
from PIL import Image
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_path = "./imgs/dog3.png"
image = Image.open(image_path)
print(image)
# png格式是四个通道，调用image.convert()函数保留颜色通道
# 若图片本来就是三个颜色通道，经此操作不变。加上这一步后，可以适应png，jpg各种格式图片
# 本案例可以不用加这个函数，因为不同截图软件截图保留的通道数不一样
image = image.convert("RGB")

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

# 搭建神经网络
class Huai(nn.Module):
    def __init__(self):
        super(Huai, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load("./huaixu_gpu_29.pth",
                   map_location=device, weights_only=False)
print(model)

image = torch.reshape(image, (1, 3, 32, 32))
image = image.to(device)

model.eval()
with torch.no_grad():  # 容易遗忘
    output = model(image)
print(output)

print(output.argmax(1))
