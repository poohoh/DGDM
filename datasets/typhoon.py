import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from PIL import Image, ImageFilter
import glob
import random

class TLoader(Dataset):
    """
    is_train - 0 train, 1 val
    """
    def __init__(self, is_train=0, path='crop_typhoon/', transform=None):
        self.is_train = is_train
        self.transform = transform
        self.path = glob.glob(f"{path}/*") * 50
        self.totenosr = transforms.ToTensor()

    def __len__(self):
        return len(self.path)

    def _make_sqe(self, path):
        if self.is_train:
            images = glob.glob(f"{path}/*/*")
            images = [i for i in images if 'ir105' in i]
            images = sorted(images)
            s_index = np.random.randint(len(images)-20)
        else:
            images = glob.glob(f"{path}/*/*")
            images = [i for i in images if 'ir105' in i]
            images = sorted(images)
            s_index = 0 
            # s_index = np.random.randint(len(images)-20)
            # print(f's_index: {s_index}')
        return images[s_index:s_index+10], images[s_index+10:s_index+20]

    def _preprocessor(self, path, is_input=False):
        imgs = []
        for i in path:
            # load images
            # ir = np.array(np.load(i)).astype(np.float32)
            # sw = np.array(np.load(i.replace('ir105', 'sw038'))).astype(np.float32)
            # wv = np.array(np.load(i.replace('ir105', 'wv063'))).astype(np.float32)
            ir = np.array(Image.open(i).convert('L'))
            sw = np.array(Image.open(i.replace('ir105', 'sw038')).convert('L'))
            wv = np.array(Image.open(i.replace('ir105', 'wv063')).convert('L'))
            
            # normalize
            # normalized_ir = ((ir - ir.min()) / (ir.max() - ir.min())).astype(np.float32)
            # normalized_sw = ((sw - sw.min()) / (sw.max() - sw.min())).astype(np.float32)
            # normalized_wv = ((wv - wv.min()) / (wv.max() - wv.min())).astype(np.float32)
            # normalized_ir = (2 * ((ir - ir.min()) / (ir.max() - ir.min())) - 1).astype(np.float32)
            # normalized_sw = (2 * ((sw - sw.min()) / (sw.max() - sw.min())) - 1).astype(np.float32)
            # normalized_wv = (2 * ((wv - wv.min()) / (wv.max() - wv.min())) - 1).astype(np.float32)

            # # npy 데이터 각 채널별 전체 평균과 std
            # ir = ((ir - 4563.149093901573) / 1296.8070864002311)
            # sw = ((sw - 15897.115747998383) / 258.12336306653293)
            # wv = ((wv - 3808.8067947994314) / 78.65618777896614)

            # standardize
            # ir = ((ir - ir.mean(axis=0)) / ir.std(axis=0))
            # sw = ((sw - sw.mean(axis=0)) / sw.std(axis=0))
            # wv = ((wv - wv.mean(axis=0)) / wv.std(axis=0))

            img = np.stack([ir, sw, wv], axis=0)
            imgs.append(img)
        imgs = np.stack(imgs, axis=0)
        # imgs = imgs.astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs / 255.

        # if to_normal is True
        imgs = (imgs - 0.5) * 2

        return imgs

    def __getitem__(self, idx):
        clip_path = self.path[idx]
        inputs, outputs = self._make_sqe(clip_path)
        
        inputs = self._preprocessor(inputs, True)
        outputs = self._preprocessor(outputs, True)
        
        return inputs, outputs    

# save npy to png
# if __name__=='__main__':
#     import os
#     import numpy as np
#     from skimage import io
#     from tqdm import tqdm

#     # 현재 디렉토리 경로 설정
#     root_dir = "data/typhoon"

#     # 저장할 루트 디렉토리 설정 (예: 'processed' 폴더 안에 저장)
#     output_root = "data/typhoon_png"


#     # Min-Max Scaling 함수
#     def min_max_scaling(image):
#         min_val = np.min(image)
#         max_val = np.max(image)

#         # min과 max 값이 동일한 경우 (이미지가 균일한 값으로 채워진 경우)
#         if max_val - min_val == 0:
#             return np.zeros_like(image)  # 모든 값을 0으로 설정
#         scaled_image = (image - min_val) / (max_val - min_val)
#         return scaled_image


#     # npy 파일을 png로 변환하여 저장하는 함수
#     def process_and_save_npy_to_png(npy_file, output_path):
#         # npy 파일 읽기
#         image = np.load(npy_file)

#         # Min-Max Scaling 수행
#         scaled_image = min_max_scaling(image)

#         # png 형식으로 저장 (0~255 범위로 변환)
#         png_image = (scaled_image * 255).astype(np.uint8)

#         # 출력 디렉토리 생성
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)

#         # png 파일 저장
#         io.imsave(output_path, png_image)


#     # 모든 npy 파일 찾기 및 처리
#     file_count = sum(len(files) for _, _, files in os.walk(root_dir))  # Get the number of files
#     with tqdm(total=file_count) as pbar:  # Do tqdm this way
#         for dirpath, _, filenames in os.walk(root_dir):
#             for filename in tqdm(filenames):
#                 if filename.endswith(".npy"):
#                     # npy 파일 경로
#                     npy_file = os.path.join(dirpath, filename)

#                     # 출력 경로 (루트 디렉토리만 변경)
#                     relative_path = os.path.relpath(npy_file, root_dir)
#                     output_path = os.path.join(output_root, os.path.splitext(relative_path)[0] + ".png")

#                     # npy 파일 처리 및 png 저장
#                     process_and_save_npy_to_png(npy_file, output_path)
#                     # print(output_path)
#                     pbar.update(1)

#     print("모든 npy 파일을 png로 변환 완료했습니다.")