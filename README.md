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
@article{perazzo2023directvoxgo++,
  title={DirectVoxGO++: Grid-based fast object reconstruction using radiance fields},
  author={Perazzo, Daniel and Lima, Jo{\~a}o Paulo and Velho, Luiz and Teichrieb, Veronica},
  journal={Computers \& Graphics},
  year={2023},
  publisher={Elsevier}
}
```
