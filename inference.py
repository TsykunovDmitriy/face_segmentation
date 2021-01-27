import os
import cv2
import argparse
from tqdm import tqdm

from src.mask_type import MaskType
from segmentator import FaceSegmentator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, help='path to images dir')
    parser.add_argument('--output', type=str, help='path to save mask')
    parser.add_argument('--mask_type', type=str, default="face", help='mask type from MaskType')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    paths_image = [os.path.join(args.images, name) for name in os.listdir(args.images) if not name.startswith(".")]

    seg = FaceSegmentator(device=0)

    for path in tqdm(paths_image):
        save_as = os.path.join(args.output, os.path.splitext(path.split("/")[-1])[0] + ".png")

        img = cv2.imread(path)
        mask = seg(img, MaskType.fromString(args.mask_type))

        cv2.imwrite(save_as, mask)
