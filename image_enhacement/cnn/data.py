from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, PILToTensor, ToPILImage
from dataset import DatasetFromFolder
from imgaug import augmenters as iaa
import numpy as np



def download_bsd(dest="dataset"):
    
    # output_image_dir = join(dest, "BSDS500/images")
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        # url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz" # for BSDS500 dataset
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"   # for BSDS300 dataset
        print("downloading url ", url)
        data = urllib.request.urlopen(url)
        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)
        remove(file_path)
    return output_image_dir

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])
def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])
    
def get_training_set(upscale_factor):
    # root_dir = download_bsd()
    train_dir = 'dataset/CelebAMask-HQ/images/train'
    # train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    return DatasetFromFolder(train_dir, 
                             input_transform=input_transform(crop_size, upscale_factor), 
                             target_transform=target_transform(crop_size))

def get_testing_set(upscale_factor):
    # root_dir = download_bsd()
    test_dir = 'dataset/CelebAMask-HQ/images/test'
    # test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))