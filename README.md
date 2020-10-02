# VGGVox-PyTorch
Implementing VGGVox for VoxCeleb1 dataset in PyTorch.

## Train

```
pip install -r requirements.txt
python3 train.py --dir ./Data/
```

###### Specify data dir with --dir

## Notes
- 81.79% Top-1 & 93.17 Top-5 Test-set accuracy, pretty satisfactory. Find details in [results.txt](results.txt).
- Training on the V100 takes 4 mins per epoch.

## Model
- Run `python3 vggm.py` for model architecture.
- Best model weights uploaded [VGGM300_BEST_140_81.99.pth](models/VGGM300_BEST_140_81.99.pth)

#### What i've done so far:
 - [x] **All the data preprocessed exactly as author's matlab code.** Checked and verified online on matlab
 - [x] **Random 3s cropped segments for training.**
 - [x] **Copy all hyperparameter**... LR, optimizer params, batch size from the author's net.
 - [x] **Stabilize PyTorch's BatchNorm and test variants.** Improved results by a small percentage.
 - [x] **Try onesided spectrogram input as mentioned on the author's github.**
 - [ ] ~~**Port the authors network from matlab and train.** The matlab model has 1300 outputs dimension, will test it later.~~
 - [ ] ~~**Copy weights from the matlab network and test.**~~

# References and Citations:

 - [VGGVox](https://github.com/a-nagrani/VGGVox)
 - linhdvu14's [vggvox-speaker-identification](https://github.com/linhdvu14/vggvox-speaker-identification)
 - jameslyons's [python_speech_features](https://github.com/jameslyons/python_speech_features)

 ```bibtex
@InProceedings{Nagrani17,
  author       = "Nagrani, A. and Chung, J.~S. and Zisserman, A.",
  title        = "VoxCeleb: a large-scale speaker identification dataset",
  booktitle    = "INTERSPEECH",
  year         = "2017",
}


@InProceedings{Nagrani17,
  author       = "Chung, J.~S. and Nagrani, A. and Zisserman, A.",
  title        = "VoxCeleb2: Deep Speaker Recognition",
  booktitle    = "INTERSPEECH",
  year         = "2018",
}
 ```



## VggVox-Mfcc 

直接使用mfcc进行声纹识别

```
mfcc(115,13)->(115,12)

自己修改了vggvox network :vggm_mfcc

训练
python train_mfcc.py

基于CMLR数据
train 和test top1 acc 为100%
```



# ID-Removing Network

复现论文[**Everybody’s Talkin’: Let Me Talk as You Want**](https://arxiv.org/pdf/1904.09571.pdf)

训练：

```
python train_idr.py


基于CMLR数据
训练完成10epoch, loss:10.6->2.48, top1 acc:100->9.68, top5 acc: 100->46.4,训练目标是 top1 1/11=9.1% ,top5 5/11=45.5% 这样达到ID 完全removed

```





