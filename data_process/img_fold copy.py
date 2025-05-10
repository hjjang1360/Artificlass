import os
import shutil
import pandas as pd
from PIL import Image
from torchvision import transforms

# 1) 메타데이터 로드 및 top7 스타일 언더샘플링
df = pd.read_csv('/home/work/workspace_ai/Artificlass/data_process/data/imagesinfo.csv')
top7 = df['style'].value_counts().nlargest(7).index.tolist()
df = df[df['style'].isin(top7)].reset_index(drop=True)

min_cnt = df['style'].value_counts().min()
df_balanced = (
    df
    .groupby('style', group_keys=False)
    .apply(lambda g: g.sample(n=min_cnt, random_state=42))
    .reset_index(drop=True)
)
print(min_cnt)

# 2) 클래스별로 80/10/10 split (int 기반 + 최소 1개 보장)
splits_data = {'train': [], 'val': [], 'test': []}

for style, grp in df_balanced.groupby('style'):
    grp = grp.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(grp)
    n_train = int(n * 0.8)
    print(n)
    print(n_train)
    n_val   = int(n * 0.1)
    n_test  = n - n_train - n_val

    # 최소 1개씩 확보 (n >= 3일 때만)
    if n_val == 0 and n > 2:
        n_val += 1
        n_train -= 1
    if n_test == 0 and n > 2:
        n_test += 1
        n_train -= 1

    splits_data['train'].append(grp.iloc[:n_train])
    splits_data['val'].append(  grp.iloc[n_train:n_train+n_val])
    splits_data['test'].append( grp.iloc[n_train+n_val:])

train_df = pd.concat(splits_data['train']).sample(frac=1, random_state=42).reset_index(drop=True)
val_df   = pd.concat(splits_data['val']).sample(frac=1, random_state=42).reset_index(drop=True)
test_df  = pd.concat(splits_data['test']).sample(frac=1, random_state=42).reset_index(drop=True)

# 3) 저장 경로 초기화
base_dir = "/home/work/workspace_ai/Artificlass/data_process/data/augmented_images_split_v3"
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.makedirs(base_dir, exist_ok=True)

# 4) 증강 파이프라인 (모든 split에 적용)
augment = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.RandomRotation(20),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        shear=10,
        scale=(0.8, 1.2)
    ),
    transforms.RandomHorizontalFlip(),
])
cnt_img=0
# 5) train/val/test 모두에 원본 + 3개 증강 저장
for split_name, split_df in [('train', train_df),
                             ('val',   val_df),
                             ('test',  test_df)]:
    print(f"[{split_name}] {len(split_df)} samples")
    for _, row in split_df.iterrows():
        style   = str(row['style'])
        fname   = row['filename']
        src     = os.path.join(
            "/home/work/workspace_ai/Artificlass/data_process/data/images",
            fname
        )
        img = Image.open(src).convert('RGB').resize((512,512))
        prefix = os.path.splitext(fname)[0]

        dst_dir = os.path.join(base_dir, split_name, style)
        os.makedirs(dst_dir, exist_ok=True)

        # 원본
        img.save(os.path.join(dst_dir, f"{prefix}_orig.jpg"), quality=95)
        if split_name == 'train':
            # 증강 3장
            for i in range(1, 3):
                aug = augment(img)
                aug.save(os.path.join(dst_dir, f"{prefix}_aug{i}.jpg"), quality=95)
        else:
            continue
        cnt_img+=1
        # 증강 3장
        # for i in range(1, 4):
        #     aug = augment(img)
        #     aug.save(os.path.join(dst_dir, f"{prefix}_aug{i}.jpg"), quality=95)
        # cnt_img+=1
print(cnt_img)
print("All done!")
