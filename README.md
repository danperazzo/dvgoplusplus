# DirectVoxGO++

Here is the code for our paper on DirectVoxGO++

### Installation
Use the docker image

```
sudo docker build -t dvgopp .
sudo nvidia-docker run -it --rm --volume /:/host --workdir /host$PWD dvgopp
```



## GO
To download the dataset, go to https://drive.google.com/file/d/1cCU35cIur-PXPKRqQxD6aKHWQmuEanoN/view?usp=share_link

### Reproduction
All config files to reproduce our results:
```bash
$ ls configs/*
configs/llff:
africa.py  ship.py  basket.py torch.py  
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