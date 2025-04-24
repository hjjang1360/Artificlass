# import pandas as pd
# import pyarrow.parquet as pq
# import pyarrow as pa
# import os
# import glob
# import gc

# def merge_parquet_files():
#     """Merge multiple parquet files into a single parquet file"""
#     current_path = '/home/work/workspace_ai/Artificlass/data_process'
#     input_dir = os.path.join(current_path, 'data', 'parquet_batches')
#     output_file = os.path.join(current_path, 'data', 'top6_styles_merged.parquet')
    
#     # Get all parquet files in the directory
#     parquet_files = sorted(glob.glob(os.path.join(input_dir, 'top6_styles_batch_*.parquet')))
#     print(f"Found {len(parquet_files)} parquet files to merge")
    
#     # Initialize empty lists to store all data
#     all_images = []
#     all_artists = []
#     all_styles = []
#     all_titles = []
    
#     # Read each parquet file and append its data
#     for i, file_path in enumerate(parquet_files):
#         print(f"Reading file {i+1}/{len(parquet_files)}: {os.path.basename(file_path)}")
#         table = pq.read_table(file_path)
        
#         # Convert to pandas for easier handling
#         df = table.to_pandas()
        
#         all_images.extend(df['image'].tolist())
#         all_artists.extend(df['artist'].tolist())
#         all_styles.extend(df['style'].tolist())
#         all_titles.extend(df['title'].tolist())
        
#         # Free memory
#         del df, table
#         gc.collect()
    
#     print(f"Successfully read all data: {len(all_images)} total samples")
    
#     # Create a new pyarrow table with all data
#     merged_table = pa.Table.from_arrays(
#         [pa.array(all_images), 
#          pa.array(all_artists), 
#          pa.array(all_styles), 
#          pa.array(all_titles)],
#         names=['image', 'artist', 'style', 'title']
#     )
    
#     # Write the merged table to a single parquet file
#     pq.write_table(merged_table, output_file)
    
#     print(f"Successfully merged all data into {output_file}")
#     print(f"Final dataset size: {len(all_images)} samples")

# if __name__ == "__main__":
#     merge_parquet_files()

import pyarrow.parquet as pq
import glob
import os
import gc

def merge_parquet_files_streaming():
    current_path = '/home/work/workspace_ai/Artificlass/data_process'
    input_dir = os.path.join(current_path, 'data', 'parquet_batches')
    output_file = os.path.join(current_path, 'data', 'top6_styles_merged.parquet')

    parquet_files = sorted(glob.glob(os.path.join(input_dir, 'top6_styles_batch_*.parquet')))
    print(f"Found {len(parquet_files)} parquet files to merge")

    # 첫 파일에서만 스키마 추출
    first_table = pq.read_table(parquet_files[0], columns=['image','artist','style','title'])
    writer = pq.ParquetWriter(output_file, schema=first_table.schema)

    # 첫 파일 쓰기
    writer.write_table(first_table)
    del first_table
    gc.collect()

    # 나머지 파일을 순차적으로 읽어 쓰기
    for i, file_path in enumerate(parquet_files[1:], start=2):
        print(f"Writing file {i}/{len(parquet_files)}: {os.path.basename(file_path)}")
        table = pq.read_table(file_path, columns=['image','artist','style','title'])
        writer.write_table(table)
        del table
        gc.collect()

    writer.close()
    print(f"Successfully merged all data into {output_file}")

if __name__ == "__main__":
    merge_parquet_files_streaming()
