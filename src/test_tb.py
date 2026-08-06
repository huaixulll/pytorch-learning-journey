from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")

image_path = "dataset/val/bees/6a00d8341c630a53ef00e553d0beb18834-800wi.jpg"

img_PIL = Image.open(image_path)

img_array = np.array(img_PIL)

print(type(img_array))
print(img_array.shape)

writer.add_image("test", img_array, 2, dataformats='HWC')

# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

writer.close()
