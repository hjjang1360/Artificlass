# Image path: /home/work/workspace_ai/Artificlass/data_process/data/images

import os
import shutil
import pandas as pd
from PIL import Image
from torchvision import transforms

# 1. 불러올 메타데이터
df = pd.read_csv('/home/work/workspace_ai/Artificlass/data_process/data/imagesinfo.csv')
top7 = df['style'].value_counts().nlargest(7).index.tolist()
df = df[df['style'].isin(top7)].reset_index(drop=True)
# df_balanced['style'] = df_balanced['style'].astype(str)
min_cnt = df['style'].value_counts().min()
df_balanced = (
    df
    .groupby('style', group_keys=False)
    .apply(lambda g: g.sample(n=min_cnt, random_state=42))
    .reset_index(drop=True)
)

# 2. 저장할 경로 만들기
# augmented_dir = './augmented_images'
augmented_dir = "/home/work/workspace_ai/Artificlass/data_process/data/augmented_images_4"
if os.path.exists(augmented_dir):
    shutil.rmtree(augmented_dir)
os.makedirs(augmented_dir, exist_ok=True)

# 3. 증강 파이프라인 정의
augment = transforms.Compose([
    # 랜덤 회전 ±20°
    transforms.RandomRotation(20),
    # ±10% 이동, shear 10°, zoom 0.8–1.2배
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        shear=10,
        scale=(0.8, 1.2)
    ),
    # 좌우 뒤집기
    transforms.RandomHorizontalFlip(),
])

# 4. 이미지 증강 및 저장
for _, row in df_balanced.iterrows():
    label    = str(row['style'])
    filename = row['filename']
    # img_path = os.path.join('./painter_by_numbers/images', filename)
    img_path=os.path.join("/home/work/workspace_ai/Artificlass/data_process/data/images", filename)

    # 클래스별 폴더
    save_dir = os.path.join(augmented_dir, label)
    os.makedirs(save_dir, exist_ok=True)

    # 원본 이미지 로드 → RGB → 512x512 리사이즈
    img = Image.open(img_path).convert('RGB')
    img = img.resize((512, 512))

    prefix = os.path.splitext(filename)[0]
    orig_path = os.path.join(save_dir, f"{prefix}_aug0.jpeg")
    img.save(orig_path, format='JPEG', quality=95)
    
    # 원본 1장 + 증강 3장
    # (원본을 저장하려면 아래 save() 호출을 한 번 더 해주시면 됩니다)
    for i in range(3):
        aug_img = augment(img)  # PIL.Image 반환
        save_path = os.path.join(save_dir, f"{prefix}_aug{i+1}.jpeg")
        aug_img.save(save_path, format='JPEG', quality=95)

print("Complete Image Augmentation")
