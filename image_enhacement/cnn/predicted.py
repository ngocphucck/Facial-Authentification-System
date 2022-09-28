import torch
from PIL import Image
from torchvision.transforms import ToTensor
from model import Net
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
upscale_factor = 3
model = Net(upscale_factor).to(device)
model = torch.load('model_epoch_10_Upscale_3.pth', map_location=device)


def predict(input_image,save_path):
    img = Image.open(input_image).convert('YCbCr')
    y, cb, cr = img.split()
    img_to_tensor = ToTensor()
    
    input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0]).to(device)
    
    out = model(input)
    out = out.cpu()
    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)

    out_ycbr = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr])
    out_ycbr.show()
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    out_img.save(save_path)
    
    return out_img