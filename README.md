# Shallow- and Deep-fake Image Manipulation Localization Using Deep Learning

## Datasets

### Deepfakes

The deepfake dataset we constructed in Section II.C of our paper can be downloaded [here](https://www.dropbox.com/s/o5410tl5v4vxsth/ICNC2023-Deepfakes.tar.xz?dl=0).

### Shallowfakes

Shallowfake dataset used in our paper can be downloaded individually via the following links:

- [CASIAv2](https://github.com/namtpham/casia2groundtruth)
- [CASIAv1](https://github.com/namtpham/casia1groundtruth)
- [Columbia](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/)
- [COVERAGE](https://github.com/wenbihan/coverage)

### Train/Val/Test Subsets

The format of each line in these files is as the following. For authentic images, `/path/to/mask.png` and `/path/to/egde.png` are set to string `None`. We use digit `0` to represent authentic images, and `1` to represent manipulated images.

```
/path/to/image.png /path/to/mask.png /path/to/egde.png 0/1
```

### backbone model achitectore :

![](./github/network.png)

### ore Custem Classifier :

![](./github/.png)

## Usage

### Training

Run the following code to train the network.

For the option `--model`, to reproduce experiments in Table III of our paper:

- Use `mvssnet` for experiments 1/2/3;
- Use `upernet` for experiments 4/5/6;
- Use `ours` for experiments 7/8/9.

```
python -u train_torch.py --paths_file /path/to/train.txt --val_paths_file /path/to/val.txt --model {mvssnet, upernet, ours}
```

### Testing

Run the following code to evaluate the network.

Trained models for experiments in Table III of our paper can be found in the following links: [1](https://www.dropbox.com/s/jov5nsj47pyfv16/1.pth?dl=0) | [2](https://www.dropbox.com/s/w9eviamadmc0feh/2.pth?dl=0) | [3](https://www.dropbox.com/s/4pq92dmjzepi0uk/3.pth?dl=0) | [4](https://www.dropbox.com/s/i9eakxvww8vsbh7/4.pth?dl=0) | [5](https://www.dropbox.com/s/0jx8pxq1aksir18/5.pth?dl=0) | [6](https://www.dropbox.com/s/adsvglkcwv6ttnj/6.pth?dl=0) | [7](https://www.dropbox.com/s/nr81w432k9llztc/7.pth?dl=0) | [8](https://www.dropbox.com/s/g2n58undkom78tb/8.pth?dl=0) | [9](https://www.dropbox.com/s/zzk4eump5xfbqmz/9.pth?dl=0).
```
python -u evaluate.py --paths_file /path/to/test.txt --load_path /path/to/trained/model.path --model {mvssnet, upernet, ours}
```
