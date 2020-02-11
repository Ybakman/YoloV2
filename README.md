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
and run
```
julia YoloGit.jl accuracy 
```
if you want to save Voc dataset output
run
```
julia YoloGit.jl accuracy --record true
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

Fill the Input folder with jpg images. The program prepares output and put them into output folder. If you want you can save output into another folder.

run the code by
```
julia YoloGit.jl saveout --record true
```
If you want to display one image and save it, you can run
```
julia YoloGit.jl display --record true
```
To change thresholds, look at the guide by
```
julia YoloGit.jl --help
```
## Example Input and Output
Here is an example of input and output:

INPUT:
<p align="center">
  <img src="dog.jpg" width="416" height="416">
</p> 

OUTPUT:
<p align="center">
  <img src="show.png" width="416" height="416">
</p> 

## Asymmetric Padding
An alternative solution for asymmetric padding on pooling.
We solved the asymmetric padding process in the following 4 steps:
Let's assume we have a matrix as (d1,d2,depth,minibatch_size)

```
1- Apply symmetric padding = 1 with the pooling function

2- reshape it into => (d1,d2,1,minibatch_size*depth)

3- apply the following convolutional layer => 1 0  with stride = 1, padding = 0
                                              0 0
                                              
4- Lastly, reshape again into => (d1,d2,depth,minibatch_size)

Now, we made pooling with asymmetric padding on the right and bottom side. If you want to 
make it on left and top side, apply this conv layer =>  0 0 
                                                        0 1

The last 3 steps are implemented in YoloPad(x) function
The symmetric padding part is implemented in the pooling function just before Yolopad(x) 
```
