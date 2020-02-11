#Author: Yavuz Faruk Bakman
#Date: 15/08/2019

#process the input and save into given directory
saveOut(model,data,confth,iouth,res,number; record = true, location = "Output") = (saveOut!(model,args,confth,iouth,res,number; record = record, location = location) for args in data)

function saveOut!(model,args,confth,iouth,res,number; record = true, location = "Output")
    out = model(args[1])
    out = postprocessing(out,confth,iouth)
    a = out
    push!(res,a)
    im = args[2][1]
    r1 = length(axes(im)[1][1:end])
    r2 = length(axes(im)[2][1:end])
    p2 = 0
    p1 = 0
    if r1> r2
        p1 = size(args[1])[1] - Int32(r2 / (r1 / size(args[1])[1]))

    end
    if r2> r1
        p2 =  size(args[1])[2] - Int32(r1 / (r2 / size(args[1])[2]))
    end
    padding = (p1,p2)
    for i in 1:length(a)
        norm = convertnormal(a,i,size(args[1]),size(args[2][1]),padding)
        drawsquare(im,norm[1],norm[2],norm[3],norm[4])
        FreeTypeAbstraction.renderstring!(im, string(numsdic[a[i][5]]), face, (14,14)  ,Int32(round(norm[2])),Int32(round(norm[1])),halign=:hleft,valign=:vtop,bcolor=eltype(im)(1.0,1.0,1.0),fcolor=eltype(im)(0,0,0)) #use `nothing` to make bcolor transparent
    end
    number[1] = number[1] + 1
    num = number[1]
    if record
        if !isdir(location)
            mkdir(location)
        end
        save(string(location,"/$num.jpg"),im)
    end
end
#confth => confidence score threshold. 0.3 is recommended
#iouth => intersection over union threshold. if 2 images overlap more than this threshold, one of them is removed
function saveoutput(model,data,confth,iouth; record = true, location = "Output")
    res = []
    number = [0]
    println("Processing Input and Saving...")
    progress!(saveOut(model,data,confth,iouth,res,number; record = record, location = location))
    println("Saved all output")
    return res
end

#draw square to given image
function drawsquare(im,x,y,w,h)
    x = Int32(round(x))
    y = Int32(round(y))
    w= Int32(round(w))
    h = Int32(round(h))

    draw!(im, LineSegment(Point(x,y), Point(x+w,y)))
    draw!(im, LineSegment(Point(x,y), Point(x,y+h)))
    draw!(im, LineSegment(Point(x+w,y), Point(x+w,y+h)))
    draw!(im, LineSegment(Point(x,y+h), Point(x+w,y+h)))
end

#Calculates accuracy for Voc Dataset
acc(model,data,confth,iouth,iou,predictions) =(acc!(model,args,confth,iouth,iou,predictions) for args in data)

function acc!(model,args,confth,iouth,iou,predictions)
    out = model(args[1])
    out = postprocessing(out,confth,iouth)
    check = zeros(length(args[2][1])-2)
    sort!(out,by = x-> x[6],rev=true)
    for k in 1:length(out)
        tp,loc = istrue(out[k],args[2][1][3:length(args[2][1])],check,iou)
        push!(predictions[numsdic[out[k][5]]],(tp,out[k][6]))
        if tp
            check[loc] = 1
        end
    end
end
#confth => confidence score threshold. 0.0 for calculating accuracy
#iouth => intersection over union threshold. if 2 images overlap more than this threshold, one of them is removed
#iou => intersection over union. True positive threshold
function accuracy(model,data,confth,iouth,iou)
    predictions = Dict("aeroplane"=>[],"bicycle"=>[],"bird"=>[], "boat"=>[],
                    "bottle"=>[],"bus"=>[],"car"=>[],"cat"=>[],"chair"=>[],
                    "cow"=>[],"diningtable"=>[],"dog"=>[],"horse"=>[],"motorbike"=>[],
                    "person"=>[],"pottedplant"=>[],"sheep"=>[],"sofa"=>[],"train"=>[],"tvmonitor"=>[])
    apdic= Dict("aeroplane"=>0.0,"bicycle"=>0.0,"bird"=>0.0, "boat"=>0.0,
                    "bottle"=>0.0,"bus"=>0.0,"car"=>0.0,"cat"=>0.0,"chair"=>0.0,
                    "cow"=>0.0,"diningtable"=>0.0,"dog"=>0.0,"horse"=>0.0,"motorbike"=>0.0,
                    "person"=>0.0,"pottedplant"=>0.0,"sheep"=>0.0,"sofa"=>0.0,"train"=>0.0,"tvmonitor"=>0.0)

    println("Calculating accuracy...")
    progress!(acc(model,data,confth,iouth,iou,predictions))
    for key in keys(predictions)
        sort!(predictions[key], by = x ->x[2],rev = true)
        tp = 0
        fp = 0
        total = totaldic[key]
        preRecall = []
        p = predictions[key]
        for i in 1:length(p)
            if p[i][1]
                tp = tp+1
            else
                fp = fp+1
            end
            if total==0
                push!(preRecall,[0,0])
            end
            push!(preRecall,[tp/(tp+fp),tp/total])
        end
        #smooth process
        rightMax = preRecall[length(preRecall)][1]
        location = length(preRecall)-1
        while(location >= 1)
            if preRecall[location][1] > rightMax
                rightMax = preRecall[location][1]
            else
                preRecall[location][1] = rightMax
            end
            location = location -1
        end
        #make calculation
        sum = 0
        for i in 2:length(preRecall)
            sum = sum + (preRecall[i][2]-preRecall[i-1][2]) * preRecall[i][1]
        end
            apdic[key] = sum
    end
    println("Calculated")
    return apdic
end

#Checks if given prediction is true positive or false negative
function istrue(prediction,labels,check,iou)
    min = iou
    result = false
    location = length(labels) + 1
    for i in 1:length(labels)
        if prediction[5] == namesdic[labels[i][5]] && check[i] == 0 && ioumatch(prediction[1],prediction[2],prediction[3],prediction[4],labels[i][1],labels[i][2],labels[i][3],labels[i][4]) > min
            min = ioumatch(prediction[1],prediction[2],prediction[3],prediction[4],labels[i][1],labels[i][2],labels[i][3],labels[i][4])
            result = true
            location = i
        end
    end
    return result ,location
end

function calculatemean(dict)
    sum = 0
    number = 0
    for key in keys(dict)
        sum = sum + dict[key]
        number = number + 1
    end
    return sum/number
end

#Displays an image's output on IDE
function displaytest(file,model; record = false)
    im, img_size, img_originalsize, padding,imgOrg = loadprepareimage(file,(416,416))
    println(img_size)
    println(img_originalsize)
    imgOrg = Array{RGB4{Float64},2}(imgOrg)
    im_input = Array{Float32}(undef,416,416,3,1)
    im_input[:,:,:,1] = permutedims(collect(channelview(im)),[2,3,1]);
    if gpu() >= 0 im_input = KnetArray(im_input) end
    res = model(im_input)
    a = postprocessing(res,0.3,0.3)
    for i in 1:length(a)
        norm = convertnormal(a,i,img_size,img_originalsize,padding)
        drawsquare(imgOrg,norm[1],norm[2],norm[3],norm[4])
        FreeTypeAbstraction.renderstring!(imgOrg, string(numsdic[a[i][5]]), face, (14,14)  ,Int32(round(norm[2])),Int32(round(norm[1])),halign=:hleft,valign=:vtop,bcolor=eltype(im)(1.0,1.0,1.0),fcolor=eltype(im)(0,0,0)) #use `nothing` to make bcolor transparent
    end
    p1 = padding[1]
    p2 = padding[2]
    display(imgOrg)
    if record save("outexample.jpg",imgOrg) end
end

function convertnormal(a,i,imgsize,img_originalsize,padding)
    x = (a[i][1] -padding[1])*img_originalsize[1]/imgsize[1]
    y = (a[i][2] -padding[2])*img_originalsize[2]/imgsize[2]
    w  = a[i][3] * img_originalsize[1]/imgsize[1]
    h = a[i][4] * img_originalsize[2]/imgsize[2]
    println(x)
    println(y)
    println(w)
    println(h)
    return x,y,w,h
end
