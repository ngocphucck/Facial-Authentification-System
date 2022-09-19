import os
import cv2
import torch
import sys
sys.path.append('.')
from image_enhacement.configs import config
import image_enhacement.utils.utils as imgproc
from image_enhacement.models.model import Generator

model = Generator().to(device=config.device, memory_format=torch.channels_last)
# Load the super-resolution model weights
checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint["state_dict"])

# Start the verification mode of the model.
model.eval()

def predict(file_path):

    lr_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    # Convert BGR channel image format data to RGB channel image format data
    lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

    # Convert RGB channel image format data to Tensor channel image format data
    lr_tensor = imgproc.image_to_tensor(lr_image, False, False).unsqueeze_(0)

    # Transfer Tensor channel image format data to CUDA device
    lr_tensor = lr_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

    # Only reconstruct the Y channel image data.
    with torch.no_grad():
        sr_tensor = model(lr_tensor)

        # Save image
    sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    name = file_path.split("/")[-1]
    cv2.imwrite(os.path.join("data/demo/enhancement/", name), sr_image)

if __name__ == "__main__":
    predict("data/demo/detection/1.jpg")