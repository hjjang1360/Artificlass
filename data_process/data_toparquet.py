# import pandas as pd
# import numpy as np
# import pyarrow.parquet as pq
# import pyarrow as pa
# from PIL import Image
# import matplotlib.pyplot as plt
# from multiprocessing import Process

# current_path='/home/work/workspace_ai/Artificlass/data_process'
# def image_load_here(image_path):
#     pass 

# if __name__ == '__main__':
#     csv_file=current_path+'/data/imagesinfo.csv'

#     data=pd.read_csv(csv_file)

#     image_path=current_path+'/data/images/'+data['filename'].values
#     artist_name=data['artist'].values
#     style_name=data['style'].values
#     title=data['title'].values

#     # Load the image data with batch size 1000 
#     batch_size=1000
#     image_size=256

#     image=np.empty((0, 3,image_size, image_size), dtype=np.uint8)
#     artist_names=np.empty((0,), dtype=object)
#     style_names=np.empty((0,), dtype=object)
#     title_names=np.empty((0,), dtype=object)

#     for i in range(0, len(image_path), batch_size):
#         p=Process(target=image_load_here, args=(image_path[i:i+batch_size]))
#         p.start()
#         p.join()
#         # The type of image is not clear, so we need to check the type of image
#         # print(type(p))

# import pandas as pd
# import numpy as np
# import pyarrow.parquet as pq
# import pyarrow as pa
# from PIL import Image
# import matplotlib.pyplot as plt
# from multiprocessing import Process, Queue
# import os
# import math
# from collections import Counter

# current_path='/home/work/workspace_ai/Artificlass/data_process'

# def image_load_here(image_paths, queue):
#     """Load images and return them as numpy arrays"""
#     batch_images = []
#     for path in image_paths:
#         try:
#             img = Image.open(path)
#             img = img.resize((256, 256))
#             img_array = np.array(img.convert('RGB'))
#             img_array = img_array.transpose(2, 0, 1)  # Convert to (C, H, W) format
#             batch_images.append(img_array)
#         except Exception as e:
#             print(f"Error loading {path}: {e}")
#             # Add a placeholder for failed images
#             batch_images.append(np.zeros((3, 256, 256), dtype=np.uint8))
    
#     queue.put(np.array(batch_images, dtype=np.uint8))

# if __name__ == '__main__':
#     csv_file=current_path+'/data/imagesinfo.csv'

#     data=pd.read_csv(csv_file)
    
#     # Find the top 6 most frequent art styles
#     style_counts = Counter(data['style'].values)
#     top_6_styles = [style for style, _ in style_counts.most_common(6)]
#     print(f"Selected top 6 styles: {top_6_styles}")
    
#     # Filter data to include only top 6 styles
#     filtered_data = data[data['style'].isin(top_6_styles)]
#     print(f"Original dataset size: {len(data)}, Filtered dataset size: {len(filtered_data)}")
    
#     # Reset filtered data indices
#     filtered_data = filtered_data.reset_index(drop=True)

#     image_path=np.array([os.path.join(current_path, 'data/images', filename) for filename in filtered_data['filename'].values])
#     artist_name=filtered_data['artist'].values
#     style_name=filtered_data['style'].values
#     title=filtered_data['title'].values

#     # Load the image data with batch size 1000 
#     batch_size=1000
#     image_size=256
#     num_batches = 10
    
#     # Calculate total items per batch file (dividing data into 10 equal parts)
#     total_items = len(image_path)
#     items_per_batch_file = math.ceil(total_items / num_batches)
    
#     # Create output directory if it doesn't exist
#     output_dir = os.path.join(current_path, 'data', 'parquet_batches')
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Process and save data in 10 batches
#     for batch_num in range(num_batches):
#         start_idx = batch_num * items_per_batch_file
#         end_idx = min(start_idx + items_per_batch_file, total_items)
        
#         if start_idx >= total_items:
#             break
            
#         print(f"Processing batch file {batch_num+1}/{num_batches} (items {start_idx} to {end_idx-1})")
        
#         # Initialize arrays for this batch file
#         image = np.empty((0, 3, image_size, image_size), dtype=np.uint8)
#         artist_names = np.empty((0,), dtype=object)
#         style_names = np.empty((0,), dtype=object)
#         title_names = np.empty((0,), dtype=object)
        
#         # Process images in smaller processing batches
#         for i in range(start_idx, end_idx, batch_size):
#             end_i = min(i + batch_size, end_idx)
#             print(f"  Processing sub-batch {(i-start_idx)//batch_size + 1} ({i} to {end_i-1})")
            
#             queue = Queue()
#             p = Process(target=image_load_here, args=(image_path[i:end_i], queue))
#             p.start()
#             batch_images = queue.get()
#             p.join()
            
#             # Append the batch data
#             image = np.vstack([image, batch_images])
#             artist_names = np.append(artist_names, artist_name[i:end_i])
#             style_names = np.append(style_names, style_name[i:end_i])
#             title_names = np.append(title_names, title[i:end_i])
        
#         # Save this batch to parquet
#         table = pa.Table.from_arrays(
#             [pa.array(image.tolist()), 
#              pa.array(artist_names.tolist()), 
#              pa.array(style_names.tolist()), 
#              pa.array(title_names.tolist())],
#             names=['image', 'artist', 'style', 'title']
#         )
        
#         output_file = os.path.join(output_dir, f'top6_styles_batch_{batch_num+1}.parquet')
#         pq.write_table(table, output_file)
#         print(f"Saved batch file {batch_num+1} to {output_file}")
        
#         # Clear memory
#         del image, artist_names, style_names, title_names, table
        
#     print("All batch files with top 6 styles successfully saved to parquet format")


import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from PIL import Image
from multiprocessing import Process, Queue
import os
import math
from collections import Counter

current_path = '/home/work/workspace_ai/Artificlass/data_process'

# -----------------------------------------------------------------------------
# Helper function to load images in parallel
# -----------------------------------------------------------------------------
def image_load_here(image_paths, queue):
    """Load images and return them as numpy arrays"""
    batch_images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            img = img.resize((256, 256))
            img_array = np.array(img.convert('RGB'))
            img_array = img_array.transpose(2, 0, 1)  # (C, H, W)
            batch_images.append(img_array)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # placeholder for failed loads
            batch_images.append(np.zeros((3, 256, 256), dtype=np.uint8))
    queue.put(np.array(batch_images, dtype=np.uint8))

# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    csv_file = os.path.join(current_path, 'data/imagesinfo.csv')
    data = pd.read_csv(csv_file)

    # 1) Identify top 6 styles
    style_counts = Counter(data['style'].values)
    top_6_styles = [s for s, _ in style_counts.most_common(6)]
    print(f"Selected top 6 styles: {top_6_styles}")

    # 2) Filter to top-6 styles only
    filtered = data[data['style'].isin(top_6_styles)].reset_index(drop=True)
    print(f"Original size: {len(data)}, filtered size: {len(filtered)}")

    # 3) Undersample each style to the minimum class count
    counts = filtered['style'].value_counts()
    min_count = counts.min()
    print(f"Undersampling each style to {min_count} samples.")
    undersampled = (
        filtered
        .groupby('style', group_keys=False)
        .apply(lambda grp: grp.sample(n=min_count, random_state=42))
        .reset_index(drop=True)
    )
    # shuffle globally
    undersampled = undersampled.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"After undersampling total size: {len(undersampled)}")

    # Prepare arrays
    image_paths = undersampled['filename'].apply(
        lambda fn: os.path.join(current_path, 'data/images', fn)
    ).values
    artists = undersampled['artist'].values
    styles = undersampled['style'].values
    titles = undersampled['title'].values

    # 4) Batch setup
    num_batches = 1
    total_items = len(image_paths)
    items_per_batch = math.ceil(total_items / num_batches)
    output_dir = os.path.join(current_path, 'data/parquet_batches')
    os.makedirs(output_dir, exist_ok=True)

    # 5) Process and write batches
    for batch_num in range(num_batches):
        start = batch_num * items_per_batch
        end = min(start + items_per_batch, total_items)
        if start >= total_items:
            break
        print(f"Batch {batch_num+1}/{num_batches}: items {start} to {end-1}")

        # initialize collectors
        imgs = np.empty((0, 3, 256, 256), dtype=np.uint8)
        art = np.empty((0,), dtype=object)
        sty = np.empty((0,), dtype=object)
        tit = np.empty((0,), dtype=object)

        # process in sub-batches if needed
        sub_batch_size = 1000
        for i in range(start, end, sub_batch_size):
            j = min(i + sub_batch_size, end)
            print(f"  sub-batch {(i-start)//sub_batch_size+1}: {i} to {j-1}")
            q = Queue()
            p = Process(target=image_load_here, args=(image_paths[i:j], q))
            p.start()
            batch_imgs = q.get()
            p.join()

            imgs = np.vstack([imgs, batch_imgs])
            art = np.append(art, artists[i:j])
            sty = np.append(sty, styles[i:j])
            tit = np.append(tit, titles[i:j])

        # write parquet
        table = pa.Table.from_arrays([
            pa.array(imgs.tolist()),
            pa.array(art.tolist()),
            pa.array(sty.tolist()),
            pa.array(tit.tolist()),
        ], names=['image', 'artist', 'style', 'title'])

        out_file = os.path.join(output_dir, f'top6_styles_batch_{batch_num+1}_under.parquet')
        pq.write_table(table, out_file)
        print(f"Saved: {out_file}")

        # cleanup
        del imgs, art, sty, tit, table
        gc.collect()

    print("Undersampled batches successfully written.")
