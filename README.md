# YoloV2
Tiny-YoloV2 implementation on Julia by Knet framework.

The program uses pre-trained weigths and currently does not train the model.

first download pre-trained weights by:
```
$ wget https://pjreddie.com/media/files/yolov2-tiny-voc.weights
```
If you want to see accuracy on Voc Dataset 2007, download the dataset by:
```
$ wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
  tar xf VOCtrainval_06-Nov-2007.tar
```
If you want to use Voc Dataset 2012
```
$ wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_11-May-2012.tar
```
And change following constants as :
```
ACC_INPUT = "VOCdevkit/VOC2012/JPEGImages"
ACC_OUT =   "VOCdevkit/VOC2012/Annotations"
```

If you don't want to see accuracy, do not download Voc-dataset and you can just comment out following lines in the code:
```
images,labels = inputandlabelsdir(ACC_OUT,ACC_INPUT)
in,out,imgs = prepareinputlabels(images,labels)

accdata = minibatch(in,out,MINIBATCH_SIZE;xtype = xtype)
drawdata = minibatch(in,imgs,MINIBATCH_SIZE; xtype = xtype)

@time AP = accuracy(model,accdata,0.0,0.3,0.5)
@time result = saveoutput(model,drawdata,0.3,0.3; record = true, location = "VocResult")
display(AP)
```
Fill the Input folder with jpg images. The program prepares output and put them into output folder. If you want you can save output into another folder.

Lastly, run the code by
```
julia YoloGit.jl
```

The code can:

1-Calculate accuracy on Voc Dataset

2-Tak the images and saves the output into the folder

3-Take an example image and diplay the output on the IDE.

Here is an example of input and output:

INPUT:
<p align="center">
  <img src="dog.jpg" width="416" height="416">
</p> 

OUTPUT:
<p align="center">
  <img src="dogout.png" width="416" height="416">
</p> 

An alternative solution for asymmetric padding on pooling.
We solved the asymmetric padding process in the following 4 steps:
Let's assume we have a matrix as (d1,d2,depth,minibatch_size)

```
1- Apply symmetric padding = 1 with the pooling function

2- reshape it into => (d1,d2,1,minibatch_size*depth)

3- apply the following convolutional layer => 1 0  with stride = 1, padding = 0
                                              0 0
                                              
4- Lastly, reshape again into => (d1,d2,depth,minibatch_size)

Now, we made pooling with asymmetric padding on the right and bottom side. If you want to make it on left and top side
apply this conv layer =>  0 0 
                          0 1

The last 3 steps are implemented in YoloPad(x) function
The symmetric padding part is implemented in the pooling function just before Yolopad(x) 
```
