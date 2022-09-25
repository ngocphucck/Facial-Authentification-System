import argparse
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
# from scipy.linalg import sqrtm
# from numpy import trace , iscomplexobj

# Training settings
parser = argparse.ArgumentParser(description='Generate Super Resolution of Input Image')
parser.add_argument('--input_image', type=str, required=True, help='Input Image filename')
parser.add_argument('--model', type=str, required=True, help='Trained Model filename')
parser.add_argument('--output_filename', type=str, help='Output Image Name')
# parser.add_argument('--cuda', action='store_true', help='To use Cuda')
opt = parser.parse_args()

print(opt)
img = Image.open(opt.input_image).convert('YCbCr')
# img.show()
y, cb, cr = img.split()

print("Loading the given model...\n")
model = torch.load(opt.model).to('cpu')
print("\nOutput is generating....\n")
img_to_tensor = ToTensor()
input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

# if opt.cuda:
#     model = model.cuda()
#     input = input.cuda()

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

out_img.save(opt.output_filename)

print('\nThe Output Image is Saved as ', opt.output_filename)
#python generate_super_resolution.py --input_image image_test/rose.png --model model_epoch_10_Upscale_3.pth --output_filename super_rose.png