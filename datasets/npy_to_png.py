import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def calculate_channel_stats(root_dir):
    # 합계와 합의 제곱 합을 저장할 변수
    ir_sum, ir_sum_sq = 0, 0
    sw_sum, sw_sum_sq = 0, 0
    wv_sum, wv_sum_sq = 0, 0
    total_pixel_count_ir = 0
    total_pixel_count_sw = 0
    total_pixel_count_wv = 0

    # 모든 파일 수 계산
    file_count = sum(len(files) for _, _, files in os.walk(root_dir))
    with tqdm(total=file_count) as pbar:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(".npy"):
                    npy_file = os.path.join(dirpath, filename)
                    data = np.load(npy_file).astype(np.float32)

                    if 'ir105' in filename:
                        ir_sum += np.sum(data)
                        ir_sum_sq += np.sum(data ** 2)
                        total_pixel_count_ir += data.size
                    elif 'sw038' in filename:
                        sw_sum += np.sum(data)
                        sw_sum_sq += np.sum(data ** 2)
                        total_pixel_count_sw += data.size
                    elif 'wv063' in filename:
                        wv_sum += np.sum(data)
                        wv_sum_sq += np.sum(data ** 2)
                        total_pixel_count_wv += data.size

                    pbar.update(1)

    # 평균 계산
    ir_mean = ir_sum / total_pixel_count_ir
    sw_mean = sw_sum / total_pixel_count_sw
    wv_mean = wv_sum / total_pixel_count_wv

    # 표준편차 계산
    ir_var = (ir_sum_sq / total_pixel_count_ir) - (ir_mean ** 2)
    sw_var = (sw_sum_sq / total_pixel_count_sw) - (sw_mean ** 2)
    wv_var = (wv_sum_sq / total_pixel_count_wv) - (wv_mean ** 2)

    ir_std = np.sqrt(ir_var)
    sw_std = np.sqrt(sw_var)
    wv_std = np.sqrt(wv_var)

    print("Channel Statistics:")
    print(f"IR Mean: {ir_mean}, IR Std: {ir_std}")
    print(f"SW Mean: {sw_mean}, SW Std: {sw_std}")
    print(f"WV Mean: {wv_mean}, WV Std: {wv_std}")

    return {
        'ir': {'mean': ir_mean, 'std': ir_std},
        'sw': {'mean': sw_mean, 'std': sw_std},
        'wv': {'mean': wv_mean, 'std': wv_std}
    }

def normalize_and_save_images(root_dir, stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # total num of files
    file_count = sum(len(files) for _, _, files in os.walk(root_dir))
    with tqdm(total=file_count) as pbar:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.npy'):
                    npy_file = os.path.join(dirpath, filename)
                    data = np.load(npy_file).astype(np.float32)

                    # normalize
                    if 'ir105' in filename:
                        mean = stats['ir']['mean']
                        std = stats['ir']['std']
                        channel = 'ir105'
                    elif 'sw038' in filename:
                        mean = stats['sw']['mean']
                        std = stats['sw']['std']
                        channel = 'sw038'
                    elif 'wv063' in filename:
                        mean = stats['wv']['mean']
                        std = stats['wv']['std']
                        channel = 'wv063'
                    else:
                        print('not existing channel')
                        exit()
                    
                    normalized = (data - mean) / std

                    # save path
                    relative_path = os.path.relpath(dirpath, root_dir)
                    save_dir = os.path.join(output_dir, relative_path)
                    os.makedirs(save_dir, exist_ok=True)

                    base_name = os.path.splitext(filename)[0]
                    png_file_name = f'{base_name}.png'
                    png_file = os.path.join(save_dir, png_file_name)

                    plt.imsave(png_file, normalized, cmap='gray')

                    pbar.update(1)
    
    print(f"Normalized images have been saved to {output_dir}")

if __name__ == '__main__':
    root_dir = "../data/typhoon_npy"
    output_dir = '../data/typhoon_png2'
    
    stats = calculate_channel_stats(root_dir)
    
    normalize_and_save_images(root_dir=root_dir, stats=stats, output_dir=output_dir)