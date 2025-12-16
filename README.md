<h1 align="center">IBGS: Image-Based Gaussian Splatting</h1>
<p align="center">
  <a href="">Hoang Chuong Nguyen</a>
  ·
  <a href="https://wei-mao-2019.github.io/home/">Wei Mao</a>
  ·
  <a href="https://alvarezlopezjosem.github.io/">Jose M. Alvarez</a>
  ·
  <a href="https://users.cecs.anu.edu.au/~mliu/">Miaomiao Liu</a>
</p>

<h2 align="center">NeurIPS 2025</h2>

<h3 align="center">
  <a href="https://openreview.net/pdf?id=AZLj6ObEDF">Paper</a> |
  <a href="https://hoangchuongnguyen.github.io/ibgs">Project Page</a> |
  <a href="https://drive.google.com/file/d/1zFshzLTFaka8Kem6K4gA5Uu_FX5Hz6vC/view?usp=sharing">Pretrained Models </a>
</h3>

<p align="center">
  <img src="assets/results.png" alt="IBGS results" width="75%">
</p>

## Installation

```shell
git clone https://github.com/HoangChuongNguyen/ibgs
cd ibgs
conda create -n ibgs python==3.8
conda activate ibgs
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 # change this based on your cuda version
pip install submodules/diff-plane-rasterization submodules/simple-knn
pip install -r requirements.txt
```

## Dataset
We use the standard benchmark datasets, which can be downloaded following <a href="https://github.com/graphdeco-inria/gaussian-splatting">3DGS</a>. For the Shiny dataset, we download the original data from <a href="https://nex-mpi.github.io/">Nex</a>, followed by running COLMAP to obtain the camera poses. The processed Shiny dataset used in our experiments can be downloaded using <a href="https://drive.google.com/file/d/1ZbVkpzbqJMjYyHeXi2WOL0BAdfKzP1av/view?usp=sharing">this link</a>. 

## Training and Evaluation

```shell

# Reproduce the results of the pretrained models
python exp_script.py

# Training without exposure correction
python train -s <path to data> -m output/<scene_name> --eval 
# Ex: python train.py -s dataset/mipnerf_360/flower -m output/flower -r 4 --eval

# To train with exposure correction 
python train -s <path to data> -m output/<scene_name>  --eval --exposure_compensation --enable_exposure_correction 
# Ex: python train.py -s dataset/tandt_db/tandt/train -m output/train -r 2 --eval --exposure_compensation --enable_exposure_correction

# Evaluation 
python render.py -s <path to data> -m output/<scene_name> # Optionally add --enable_exposure_correction if previously trained with exposure correction
python metrics.py -m output/<scene_name> 
```

## Citation
Please consider citing our work if you find it interesting. 

```
@inproceedings{
  nguyen2025ibgs,
  title={{IBGS}: Image-Based Gaussian Splatting},
  author={Hoang Chuong Nguyen and Wei Mao and Jose M. Alvarez and Miaomiao Liu},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=AZLj6ObEDF}
}
```

## Acknowledgement

This research was supported in part by the Australia Research Council ARC Discovery Grant (DP200102274).

We developed our method upon the code base provided by <a href="https://github.com/graphdeco-inria/gaussian-splatting">3DGS</a> and <a href="https://github.com/zju3dv/PGSR">PGSR</a>. We thank the authors for providing their excellent code. 
