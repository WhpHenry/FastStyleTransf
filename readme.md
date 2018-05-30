# [Fast Neural Style Experiment](https://arxiv.org/abs/1603.08155)

Deeply based on:
 [hzy46/fast-neural-style-tensorflow](https://github.com/hzy46/fast-neural-style-tensorflow) and [OlavHN/fast-neural-style](https://github.com/OlavHN/fast-neural-style). 
 Your kind assistance are very muchappreciated.

## Basic Enviornment and Requirement:
- Python 3.6
- Tensorflow-gpu 1.4.0

## Notice: 

- pip install pyyaml
- training image dataset: [COCO Dataset 12+G](http://msvocds.blob.core.windows.net/coco2014/train2014.zip) 
- Pretrain loss model: [VGG16 model from Slim 527M](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)  
- *.yml file path is './conf/' ,as default
- style models path is './gen_model/'
- images of style model path is './gen_model/model_img/'
- coco images dataset path is './imgs4train/'

### Train self-model:

-be sure './loss_model/vgg_16.ckpt' exists
```
python train.py <-c conf/**.yml> 
```

### Use Trained Models:
-be sure all style models have been download
-To generate a sample from the model "*.ckpt-done" and image "test.jpg", run:
```
python generate.py <--model_file *.ckpt-done> <--input_image test.jpg> <--output_image res.jpg>
```