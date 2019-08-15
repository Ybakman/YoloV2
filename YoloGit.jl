#Author: Yavuz Faruk Bakman
#Date: 15/08/2019
#Description: Tiny Yolo V2 implementation by Knet framework.
#Currently, it uses pre-trained weigths and doesn't train the model

#Load necessary packages
using Pkg; for p in ("Knet","Random","Glob","FileIO","DelimitedFiles","OffsetArrays","Images",
"ImageDraw","ImageMagick","ImageFiltering","ImageTransformations","Colors","FreeTypeAbstraction","QuartzImageIO","LightXML");
haskey(Pkg.installed(),p) || Pkg.add(p); end

xtype=(Knet.gpu()>=0 ? Knet.KnetArray{Float32} : Array{Float32})#if gpu exists run on gpu
#Include related files.
include("inputprocess.jl") #containing functions preparing input
include("outprocess.jl") # containing functions preparing output
include("loadweights.jl") # containing functions loading weights to model

using Knet: Knet, progress, progress!, gpu, KnetArray,
relu,minibatch,conv4, pool, softmax,minibatch
using Random, Glob, FileIO, DelimitedFiles, OffsetArrays
using Images, ImageDraw, ImageFiltering, ImageTransformations, Colors
using FreeTypeAbstraction
using LightXML
using ImageMagick

face = newface("DroidSansMono.ttf") #Font type

#Yolo V2 pre-trained boxes width and height
anchors = [(1.08,1.19),  (3.42,4.41),  (6.63,11.38),  (9.42,5.11),  (16.62,10.52)]
#20 classes on Voc dataset
numClass = 20

MINIBATCH_SIZE = 1
WEIGHTS_FILE = "yolov2-tiny-voc.weights" #Pre-trained weights data
ACC_INPUT = "VOCdevkit/VOC2012/JpegImages" #Input directory for accuracy calculation
ACC_OUT =   "VOCdevkit/VOC2012/Annotations" #location of objects as Xml file for accuracy calculation
INPUT =     "Input"    #Input directory to create output and save
EXAMPLE_INPUT = "example.jpg" #One input for display

# 2 dictionaries to access number<->class by O(1)
namesdic = Dict("aeroplane"=>1,"bicycle"=>2,"bird"=>3, "boat"=>4,
            "bottle"=>5,"bus"=>6,"car"=>7,"cat"=>8,"chair"=>9,
            "cow"=>10,"diningtable"=>11,"dog"=>12,"horse"=>13,"motorbike"=>14,
            "person"=>15,"pottedplant"=>16,"sheep"=>17,"sofa"=>18,"train"=>19,"tvmonitor"=>20)
numsdic =  Dict(1=>"aeroplane",2=>"bicycle",3=>"bird", 4=>"boat",
            5=>"bottle",6=>"bus",7=>"car",8=>"cat",9=>"chair",
            10=>"cow",11=>"diningtable",12=>"dog",13=>"horse",14=>"motorbike",
            15=>"person",16=>"pottedplant",17=>"sheep",18=>"sofa",19=>"train",20=>"tvmonitor")

#Store total amount of the objects to calculate accuracy
totaldic = Dict("aeroplane"=>0,"bicycle"=>0,"bird"=>0, "boat"=>0,
                "bottle"=>0,"bus"=>0,"car"=>0,"cat"=>0,"chair"=>0,
                "cow"=>0,"diningtable"=>0,"dog"=>0,"horse"=>0,"motorbike"=>0,
                "person"=>0,"pottedplant"=>0,"sheep"=>0,"sofa"=>0,"train"=>0,"tvmonitor"=>0)

#Define sigmoid function
sigmoid(x) = 1.0 / (1.0 .+ exp(-x))

#Define Chain
mutable struct Chain
    layers
    Chain(layers...) = new(layers)
end

mutable struct Conv; w; b; stride; padding; f; end #Define convolutional layer
mutable struct YoloPad; w; end #Define Yolo padding layer (assymetric padding).
struct Pool; size; stride; pad; end # Define pool layer

YoloPad(w1::Int,w2::Int,cx::Int,cy::Int) = YoloPad(zeros(Float32,w1,w2,cx,cy))#Constructor for Yolopad
Conv(w1::Int,w2::Int,cx::Int,cy::Int,st,pd,f) = Conv(randn(Float32,w1,w2,cx,cy),randn(Float32,1,1,cy,1),st,pd,f)#Constructor for convolutional layer

#Assymetric padding function
function(y::YoloPad)(x)
    x = reshape(x,14,14,1,512*MINIBATCH_SIZE)
    return reshape(conv4(y.w,x; stride = 1),13,13,512,MINIBATCH_SIZE)
end

(p::Pool)(x) = pool(x; window = p.size, stride = p.stride, padding=p.pad) #pool function
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x) #chain function
(c::Conv)(x) = c.f.(conv4(c.w,x; stride = c.stride, padding = c.padding) .+ c.b) #convolutional layer function

#leaky function
function leaky(x)
    if gpu() < 0
        return max(convert(Float32,0.1*x),x)
    end
    return max(0.1*x,x)
end

#Tiny Yolo V2 model configuration
model = Chain(Conv(3,3,3,16,1,1,leaky),
              Pool(2,2,0),
              Conv(3,3,16,32,1,1,leaky),
              Pool(2,2,0),
              Conv(3,3,32,64,1,1,leaky),
              Pool(2,2,0),
              Conv(3,3,64,128,1,1,leaky),
              Pool(2,2,0),
              Conv(3,3,128,256,1,1,leaky),
              Pool(2,2,0),
              Conv(3,3,256,512,1,1,leaky),
              Pool(2,1,1),
              YoloPad(2,2,1,1),
              Conv(3,3,512,1024,1,1,leaky),
              Conv(3,3,1024,1024,1,1,leaky),
              Conv(1,1,1024,125,1,0,identity))

#post processing function.
#Confidence score threshold to select correct predictions. Recommended : 0.3
#IoU threshold to remove unnecessary predictions: Recommended:0.3
function postprocessing(out,confth,ıouth)
    out = Array{Float32,4}(out)
    result = []
    RATE = 32
    for cy in 1:13
        for cx in 1:13
            for b in 1:5
                channel = (b-1)*(numClass + 5)
                tx = out[cy,cx,channel+1,1]
                ty = out[cy,cx,channel+2,1]
                tw = out[cy,cx,channel+3,1]
                th = out[cy,cx,channel+4,1]
                tc = out[cy,cx,channel+5,1]
                x = (sigmoid(tx) + cx-1) * RATE
                y = (sigmoid(ty) + cy-1) * RATE
                w = exp(tw) * anchors[b][1] * RATE
                h = exp(th) * anchors[b][2] * RATE
                conf = sigmoid(tc)
                classScores = Array{Float32}(UndefInitializer(),20);
                for i in channel+6:channel+25
                    classScores[i-channel-5] = out[cy,cx,i,1]
                end
                classScores = softmax(classScores)
                classNo = argmax(classScores)
                bestScore = classScores[classNo]
                classConfidenceScore = conf*bestScore
                if classConfidenceScore > confth
                     p = (max(0.0,x-w/2),max(0.0,y-h/2),min(w,416.0),min(h,416.0),classNo,classConfidenceScore)
                     push!(result,p)
                end
            end
        end
    end
    result = nonmaxsupression(result,ıouth)
    return result
end

#It removes the predictions overlapping.
function nonmaxsupression(results,ıouth)
    sort!(results, by = x ->x[6],rev=true)
    for i in 1:length(results)
        k = i+1
        while k <= length(results)
            if ıoumatch(results[i][1],results[i][2],results[i][3],results[i][4],
                results[k][1],results[k][2],results[k][3],results[k][4]) > ıouth
                deleteat!(results,k)
                k = k - 1
            end
            k = k+1
        end
    end
 return results
end

#It calculates IoU score (overlapping rate)
function ıoumatch(x1,y1,w1,h1,x2,y2,w2,h2)
        r1 = x1 + w1
        l1 = x1
        t1 = y1
        b1 = y1 + h1
        r2 = x2 + w2
        l2 = x2
        t2 = y2
        b2 = y2 + h2
        a = min(r1,r2)
        b = max(t1,t2)
        c = max(l1,l2)
        d = min(b1,b2)
        intersec = (d-b)*(a-c)
        return intersec/(w1*h1+w2*h2-intersec)
end

#Load pre-trained weights into the model
f = open(WEIGHTS_FILE)
getweights(model,f)

#=User guide
inputandlabelsdir => takes Voc labels and inputs folder location respectively and returns 2 arrays
images directories and their labels' directories.

prepareinputlabels => takes images and labels directories and returns 3 arrays
input as 416*416*3*totalImage and
labels as tupple of arrays.
tupples are designed as (ImageWidth, ImageHeight,[x,y,objectWidth,objectHeight],[x,y,objectWidth,objectHeight]..)
images are padded version of given images.

inputdir => takes the directory of input for saving output and returns array of directories of the images
prepareinput => takes the array of directories of the images and returns 416*416*3*totalImage. =#

#prepare data for accuracy
images,labels = inputandlabelsdir(ACC_OUT,ACC_INPUT)
in,out,imgs = prepareinputlabels(images,labels)

#prepare data for saving process
indir = inputdir(INPUT)
inp,images = prepareinput(indir)

#Minibatching process
accdata = minibatch(in,out,MINIBATCH_SIZE;xtype = xtype)
drawdata = minibatch(in,imgs,MINIBATCH_SIZE; xtype = xtype)
savedata = minibatch(inp,images,MINIBATCH_SIZE; xtype = xtype)

#Display one test image
displaytest(EXAMPLE_INPUT,model; record = false)

#calculate accuracy. returns a dictionary containing all classes and their average precision score.
#if overlapping is more than 0.5, prediction considered as true positive.
#if two predictions overlap more than 0.3, one of them is removed
@time AP = accuracy(model,accdata,0.0,0.3,0.5)
display(AP)

#output of Voc dataset.
#return output as [ [(x,y,width,height,classNumber,confidenceScore),(x,y,width,height,classNumber,confidenceScore)...] ,[(x,y,width,height,classNumber,confidenceScore),(x,y,width,height,classNumber,confidenceScore)..],...]
#save the output images into given location
@time result = saveoutput(model,drawdata,0.3,0.3; record = true, location = "VocResult")


#output of given input.
#return output as same with above example
#It also saves the result of the images into output folder.
@time result2 = saveoutput(model,savedata,0.3,0.3; record = true, location = "Output")

close(f)
