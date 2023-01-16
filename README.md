# DirectVoxGO++

DirectVoxGO (Direct Voxel Grid Optimization, see our [paper](https://arxiv.org/abs/2111.11215)) reconstructs a scene representation from a set of calibrated images capturing the scene.
- **NeRF-comparable quality** for synthesizing novel views from our scene representation.
- **Super-fast convergence**: Our **`15 mins/scene`** vs. NeRF's `10~20+ hrs/scene`.
- **No cross-scene pre-training required**: We optimize each scene from scratch.
- **Better rendering speed**: Our **`<1 secs`** vs. NeRF's `29 secs` to synthesize a `800x800` images.

Below run-times (*mm:ss*) of our optimization progress are measured on a machine with a single RTX 2080 Ti GPU.

https://user-images.githubusercontent.com/2712505/142961346-82cd84f5-d46e-4cfc-bce5-2bbb78f16272.mp4


### Installation
Use the docker image

```
sudo docker build -t dvgopp .
sudo nvidia-docker run -it --rm --volume /:/host --workdir /host$PWD dvgopp
```


<details>
  <summary> Dependencies (click to expand) </summary>

  - `PyTorch`, `numpy`: main computation.
  - `scipy`, `lpips`: SSIM and LPIPS evaluation.
  - `tqdm`: progress bar.
  - `mmcv`: config system.
  - `opencv-python`: image processing.
  - `imageio`, `imageio-ffmpeg`: images and videos I/O.
</details>





## GO
To download the dataset, go to https://drive.google.com/file/d/1cCU35cIur-PXPKRqQxD6aKHWQmuEanoN/view?usp=share_link

### Reproduction
All config files to reproduce our results:
```bash
$ ls configs/*
configs/llff:
africa.py  ship.py  basket.py  
```

## Acknowledgement
The code base is origined from DirectVoxGO, we thank the authors for releasing their awesome code! We are standing on the shoulders of giants!:)


## Citation
If you find this work helps you, please cite: 
```
@article{perazzodirect,
  title={DirectVoxGO++: Fast Neural Radiance Fields for Object Reconstruction},
  author={Perazzo, Daniel and Lima, Joao Paulo and Velho, Luiz and Teichrieb, Veronica},
  journal={SIBGRAPI 2022},
  year={2022}
}
```