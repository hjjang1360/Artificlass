# import pandas as pd
# import numpy as np
# import h5py
# from PIL import Image
# from multiprocessing import Process, Queue
# import os
# import math
# from collections import Counter
# import gc
# import torch
# from torchvision import transforms

# current_path = '/home/work/workspace_ai/Artificlass/data_process'

# # -----------------------------------------------------------------------------
# # Helper function to load images in parallel
# # -----------------------------------------------------------------------------
# def image_load_here(image_paths, queue):
#     """Load images and return them as numpy arrays (C,H,W uint8)"""
#     batch_images = []
#     for path in image_paths:
#         try:
#             img = Image.open(path)
#             img = img.resize((512, 512))
#             arr = np.array(img.convert('RGB'), dtype=np.uint8)
#             arr = arr.transpose(2, 0, 1)  # (C, H, W)
#         except Exception as e:
#             print(f"Error loading {path}: {e}")
#             break
#             arr = np.zeros((3, 512, 512), dtype=np.uint8)
#         batch_images.append(arr)
#     queue.put(np.stack(batch_images, axis=0))  # (N, 3, 512, 512)

# # -----------------------------------------------------------------------------
# # Main processing
# # -----------------------------------------------------------------------------
# if __name__ == '__main__':
#     csv_file = os.path.join(current_path, 'data/imagesinfo.csv')
#     data = pd.read_csv(csv_file)
#     aug_factor = 3
#     # 1) Top-7 스타일 선정
#     style_counts = Counter(data['style'])
#     top_7 = [s for s,_ in style_counts.most_common(7)]
#     print(f"Selected top 7 styles: {top_7}")

#     # 2) 필터링 & 셔플
#     filtered = data[data['style'].isin(top_7)].reset_index(drop=True)
#     min_count = filtered['style'].value_counts().min()
#     print(f"Undersample each to {min_count} samples.")
#     undersampled = (
#         filtered
#         .groupby('style', group_keys=False)
#         .apply(lambda g: g.sample(n=min_count, random_state=42))
#         .reset_index(drop=True)
#         .sample(frac=1, random_state=42)
#         .reset_index(drop=True)
#     )
#     print(f"After undersampling: {len(undersampled)} items")
    
#     # augment = transforms.Compose([
#     #     # transforms.ToPILImage(),                # numpy→PIL
#     #     transforms.RandomRotation(20),
#     #     transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
#     #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     #     transforms.RandomHorizontalFlip(),
#     #     transforms.ToTensor()                   # PIL→Tensor (C,H,W), float [0,1]
#     # ])

#     # 파일 경로와 메타데이터 준비
#     image_paths = undersampled['filename'].apply(
#         lambda fn: os.path.join(current_path, 'data/images', fn)
#     ).tolist()
#     artists = undersampled['artist'].tolist()
#     styles  = undersampled['style'].tolist()
#     titles  = undersampled['title'].tolist()

#     # 3) 배치 설정
#     num_batches     = 1
#     total_items     = len(image_paths)
#     items_per_batch = math.ceil(total_items / num_batches)
#     output_dir      = os.path.join(current_path, 'data/h5_batches')
#     os.makedirs(output_dir, exist_ok=True)

#     # 4) 각각 배치별로 HDF5 저장
#     for b in range(num_batches):
#         start = b * items_per_batch
#         end   = min(start + items_per_batch, total_items)
#         if start >= end:
#             break
#         print(f"Batch {b+1}/{num_batches}: items {start}–{end-1}")

#         # 빈 배열 준비
#         imgs = np.empty((0, 3, 512, 512), dtype=np.uint8)
#         arts = []
#         stys = []
#         tits = []

#         # sub-batch 단위로 이미지 로드
#         sub_bs = 1000
#         for i in range(start, end, sub_bs):
#             j = min(i + sub_bs, end)
#             print(f"  sub-batch {i}–{j-1}", flush=True)

#             q = Queue()
#             p = Process(target=image_load_here, args=(image_paths[i:j], q))
#             p.start()
#             batch_imgs = q.get()  # (N,3,512,512)
#             p.join()

#             aug_list = []
#             for arr in batch_imgs:
#                 # numpy CHW → PIL (HWC)
#                 pil = Image.fromarray(arr.transpose(1,2,0))
#                 for _ in range(aug_factor):
#                     t = augment(pil)  # Tensor C×H×W, float
#                     np_arr = (t.mul(255)
#                                .byte()
#                                .numpy())     # uint8 C×H×W
#                     aug_list.append(np_arr)
            
#             # imgs = np.vstack((imgs, batch_imgs))
#             # arts.extend(artists[i:j])
#             # stys.extend(styles[i:j])
#             # tits.extend(titles[i:j])
#             aug_all = np.stack(aug_list, axis=0)  # (N*aug_factor, C, H, W)
#             combined = np.vstack((batch_imgs, aug_all))  # (N+N*aug, C, H, W)
#             imgs = np.vstack((imgs, combined))

#             # (3) 메타데이터 복제
#             arts.extend(artists[i:j] * (1 + 0))  # 원본
#             arts.extend(artists[i:j] * aug_factor)  # 증강
#             stys.extend(styles[i:j] * (1 + 0))
#             stys.extend(styles[i:j] * aug_factor)
#             tits.extend(titles[i:j] * (1 + 0))
#             tits.extend(titles[i:j] * aug_factor)

#         # HDF5 파일로 쓰기
#         out_path = os.path.join(output_dir, f'top7_h5_batch{b+1}.h5')
#         with h5py.File(out_path, 'w') as f:
#             # 이미지 데이터셋: chunk 단위, gzip 압축
#             f.create_dataset(
#                 'images',
#                 data=imgs,
#                 dtype='uint8',
#                 chunks=(min(64, imgs.shape[0]), 3, 512, 512),
#                 compression='gzip',
#                 compression_opts=4
#             )
#             # 문자열 데이터셋: UTF-8 가변 길이
#             str_dt = h5py.string_dtype(encoding='utf-8')
#             f.create_dataset('artist', data=np.array(arts, dtype=object),
#                              dtype=str_dt)
#             f.create_dataset('style',  data=np.array(stys, dtype=object),
#                              dtype=str_dt)
#             f.create_dataset('title',  data=np.array(tits, dtype=object),
#                              dtype=str_dt)

#         print(f"Saved HDF5: {out_path}")

#         # 메모리 정리
#         del imgs, arts, stys, tits
#         gc.collect()

#     print("All batches written to HDF5.")

import pandas as pd
import numpy as np
import h5py
from PIL import Image
from multiprocessing import Process, Queue
import os
import math
from collections import Counter
import gc

current_path = '/home/work/workspace_ai/Artificlass/data_process'

def image_load_here(image_paths, queue):
    """이미지를 (C,H,W,uint8) 배열로 읽어 Queue에 전달."""
    batch_images = []
    for path in image_paths:
        try:
            img = Image.open(path).resize((512, 512))
            arr = np.array(img.convert('RGB'), dtype=np.uint8).transpose(2, 0, 1)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
        batch_images.append(arr)
    if batch_images:
        queue.put(np.stack(batch_images, axis=0))

if __name__ == '__main__':
    # 1) 메타데이터 로드 및 Top-7 스타일 선정
    csv_file = os.path.join(current_path, 'data/imagesinfo.csv')
    data = pd.read_csv(csv_file)
    top_7 = [s for s,_ in Counter(data['style']).most_common(7)]
    filtered = data[data['style'].isin(top_7)].reset_index(drop=True)
    min_count = filtered['style'].value_counts().min()
    undersampled = (
        filtered
        .groupby('style', group_keys=False)
        .apply(lambda g: g.sample(n=min_count, random_state=42))
        .reset_index(drop=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    # 2) 경로 & 메타 준비
    image_paths = undersampled['filename'].apply(
        lambda fn: os.path.join(current_path, 'data/images', fn)
    ).tolist()
    artists = undersampled['artist'].tolist()
    styles  = undersampled['style'].tolist()
    titles  = undersampled['title'].tolist()

    # 3) 배치 세팅
    num_batches     = 1
    total_items     = len(image_paths)
    items_per_batch = math.ceil(total_items / num_batches)
    output_dir      = os.path.join(current_path, 'data/h5_batches')
    os.makedirs(output_dir, exist_ok=True)
    sub_bs = 1000

    # 4) HDF5 생성
    for b in range(num_batches):
        start = b * items_per_batch
        end   = min(start + items_per_batch, total_items)
        if start >= end:
            break

        imgs = np.empty((0, 3, 512, 512), dtype=np.uint8)
        arts = []; stys = []; tits = []

        print(f"Batch {b+1}/{num_batches}: items {start}–{end-1}")
        for i in range(start, end, sub_bs):
            j = min(i + sub_bs, end)
            print(f"  sub-batch {i}–{j-1}")

            q = Queue()
            p = Process(target=image_load_here, args=(image_paths[i:j], q))
            p.start()
            batch_imgs = q.get()        # (N,3,512,512)
            p.join()

            imgs = np.vstack((imgs, batch_imgs))
            arts.extend(artists[i:j])
            stys.extend(styles[i:j])
            tits.extend(titles[i:j])

        # 검증
        assert imgs.shape[0] == len(arts) == len(stys) == len(tits)

        # HDF5 쓰기
        out_path = os.path.join(output_dir, f'top7_h5_batch{b+1}.h5')
        with h5py.File(out_path, 'w') as f:
            f.create_dataset(
                'images',
                data=imgs,
                dtype='uint8',
                chunks=(min(64, imgs.shape[0]), 3, 512, 512),
                compression='gzip',
                compression_opts=4
            )
            str_dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('artist', data=np.array(arts, dtype=object), dtype=str_dt)
            f.create_dataset('style',  data=np.array(stys, dtype=object), dtype=str_dt)
            f.create_dataset('title',  data=np.array(tits, dtype=object), dtype=str_dt)

        print(f"Saved HDF5: {out_path}")
        del imgs, arts, stys, tits
        gc.collect()

    print("All batches written to HDF5.")
