
import os

mip_nerf_dir = "dataset/mip_nerf_360"
tanks_and_temple_dir = "dataset/tanks"
deep_blending_dir = "dataset/deep_blending"
shiny_dir = "dataset/_SHINNY_DATASET_"

# MipNeRF360 indoor scenes
mipnerf360_indoor_scene_list = ['bonsai', 'counter', 'kitchen', 'room']
for scene in mipnerf360_indoor_scene_list:
    os.system(f"python train.py -s {mip_nerf_dir}/{scene} -m output/mip_nerf_360/{scene} -r 2 --eval")
    os.system(f"python render.py -s {mip_nerf_dir}/{scene} -m output/mip_nerf_360/{scene}  -r 2 --eval")
    os.system(f"python metrics.py -m output/mip_nerf_360/{scene}")

# # MipNeRF360 outdoor scenes
mipnerf360_outdoor_scene_list = ['bicycle', 'flowers', 'garden', 'stump', 'treehill']
for scene in mipnerf360_outdoor_scene_list:
    os.system(f"python train.py -s {mip_nerf_dir}/{scene} -m output/mip_nerf_360/{scene} -r 4 --eval")
    os.system(f"python render.py -s {mip_nerf_dir}/{scene} -m output/mip_nerf_360/{scene}  -r 4 --eval")
    os.system(f"python metrics.py -m output/mip_nerf_360/{scene}")

# # Deep blending scenes
deepblending_scene_list = ['drjohnson', 'playroom']
for scene in deepblending_scene_list:
    os.system(f"python train.py -s {deep_blending_dir}/{scene} -m output/deep_blending/{scene} -r 1 --eval  --multi_view_max_angle 50 --multi_view_max_dis 4.5 ")
    os.system(f"python render.py -s {deep_blending_dir}/{scene} -m output/deep_blending/{scene}  -r 1 --eval  --multi_view_max_angle 50 --multi_view_max_dis 4.5")
    os.system(f"python metrics.py -m output/deep_blending/{scene}")

# # Shiny scenes
shiny_scene_list = ['guitars', 'lab', 'cd']
for scene in shiny_scene_list:
    os.system(f"python train.py -s {shiny_dir}/{scene} -m output/shiny/{scene} -r 1008 --eval  --multi_view_max_angle 50 --multi_view_max_dis 4.5 ")
    os.system(f"python render.py -s {shiny_dir}/{scene} -m output/shiny/{scene}  -r 1008 --eval  --multi_view_max_angle 50 --multi_view_max_dis 4.5")
    os.system(f"python metrics.py -m output/shiny/{scene}")

# # Tanks&Temples scenes
tanks_scene_list = ['train', 'truck']
for scene in tanks_scene_list:
    os.system(f"python train.py -s {tanks_and_temple_dir}/{scene} -m output/tanks/{scene} -r 2 --eval --exposure_compensation --enable_exposure_correction")
    os.system(f"python render.py -s {tanks_and_temple_dir}/{scene} -m output/tanks/{scene}  -r 2  --eval --exposure_compensation --enable_exposure_correction")
    os.system(f"python metrics.py -m output/tanks/{scene}")
