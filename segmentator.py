import os
import cv2
import torch
import numpy as np
from os.path import dirname, abspath, join, exists

from src.model import BiSeNet
from load_url import download_file_from_google_drive
from src.mask_type import MaskType


current_dir = dirname(abspath(__file__))
id_pretrained = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"


def normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value
    return (img.astype(np.float32) - mean)/std


def to_tensor(img):
    img = normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0)


def vis_parsing_maps(im, parsing_anno, stride):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    return vis_im



mask_type_to_indexes = {
    MaskType.FACE: [1, 2, 3, 4, 5, 6, 10, 11, 12, 13], 
    MaskType.HEAD: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18],
    MaskType.ALL: np.arange(1, 19),
}


class FaceSegmentator:
    def __init__(self, device="cpu"):
        try:
            device = int(device)
            cuda_device = f"cuda:{device}"
            self.device = torch.device(cuda_device) if torch.cuda.is_available() else torch.device("cpu")
        except ValueError:
            self.device = torch.device("cpu")

        self.seg_model = BiSeNet(n_classes=19)
        self.seg_model.to(self.device)
        self.seg_model.eval()

        pretrained_path = join(current_dir, "pretrained", "bisenet.pth")
        if not exists(pretrained_path):
            os.makedirs(join(current_dir, "pretrained"), exist_ok=True)
            download_file_from_google_drive(id_pretrained, pretrained_path)
        
        self.seg_model.load_state_dict(torch.load(pretrained_path, map_location=self.device))

    def __call__(self, img, mask_type):
        scale = 512 / max(img.shape[:2])
        img_scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        img_ = np.zeros((512, 512, 3))
        img_[:img_scaled.shape[0], :img_scaled.shape[1]] = img_scaled
        img_ = to_tensor(img_)

        with torch.no_grad():
            out = self.seg_model(img_.to(self.device))[0]
        
        parsing = out.squeeze(0).cpu().numpy().argmax(0).astype(np.uint8)
        parsing = parsing[:img_scaled.shape[0], :img_scaled.shape[1]]
        parsing = cv2.resize(parsing, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        parsing = np.isin(parsing, mask_type_to_indexes[mask_type]).astype(np.uint8)

        return parsing


if __name__ == "__main__":
    s = FaceSegmentator()
    img = cv2.imread("./assets/celeb_002_2.jpg")
    res = s(img, MaskType.ALL)
    # vis_im = vis_parsing_maps(img, res, 1)
    cv2.imwrite("assets/all_mask.png", res*255)




        


