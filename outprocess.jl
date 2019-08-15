function saveoutput(model,data,confth,ıouth; record = true, location = "Output")
    res = []
    number = 0
    for (x,y) in data
        out = model(x)
        out = postprocessing(out,confth,ıouth)
        a = out
        push!(res,a)
        im = y[1]
        p2 = 416-length(axes(im)[1][1:end])
        p1 = 416-length(axes(im)[2][1:end])
        padding = (p1,p2)
        for i in 1:length(a)
            drawsquare(im,a[i][1],a[i][2],a[i][3],a[i][4],padding)
            FreeTypeAbstraction.renderstring!(im, string(numsdic[a[i][5]]), face, (14,14)  ,Int32(round(a[i][2]))-padding[2],Int32(round(a[i][1]))-padding[1],halign=:hleft,valign=:vtop,bcolor=eltype(im)(1.0,1.0,1.0),fcolor=eltype(im)(0,0,0)) #use `nothing` to make bcolor transparent
        end
        number = number + 1
        if record
            if !isdir(location)
                mkdir(location)
            end
            save(string(location,"/$number.jpg"),im[1:end-p2,1:end-p1])
        end
    end
    return res
end

function drawsquare(im,x,y,w,h,padding)
    x = Int32(round(x))-padding[1]
    y = Int32(round(y))-padding[2]
    w= Int32(round(w))
    h = Int32(round(h))

    draw!(im, LineSegment(Point(x,y), Point(x+w,y)))
    draw!(im, LineSegment(Point(x,y), Point(x,y+h)))
    draw!(im, LineSegment(Point(x+w,y), Point(x+w,y+h)))
    draw!(im, LineSegment(Point(x,y+h), Point(x+w,y+h)))
end

function accuracy(model,data,confth,ıouth,ıou)
    predictions = Dict("aeroplane"=>[],"bicycle"=>[],"bird"=>[], "boat"=>[],
                    "bottle"=>[],"bus"=>[],"car"=>[],"cat"=>[],"chair"=>[],
                    "cow"=>[],"diningtable"=>[],"dog"=>[],"horse"=>[],"motorbike"=>[],
                    "person"=>[],"pottedplant"=>[],"sheep"=>[],"sofa"=>[],"train"=>[],"tvmonitor"=>[])
    apdic= Dict("aeroplane"=>0.0,"bicycle"=>0.0,"bird"=>0.0, "boat"=>0.0,
                    "bottle"=>0.0,"bus"=>0.0,"car"=>0.0,"cat"=>0.0,"chair"=>0.0,
                    "cow"=>0.0,"diningtable"=>0.0,"dog"=>0.0,"horse"=>0.0,"motorbike"=>0.0,
                    "person"=>0.0,"pottedplant"=>0.0,"sheep"=>0.0,"sofa"=>0.0,"train"=>0.0,"tvmonitor"=>0.0)
    for (x,y) in data
        out = model(x)
        out = postprocessing(out,confth,ıouth)
        check = zeros(length(y[1])-2)
        sort!(out,by = x-> x[2],rev=true)
        for k in 1:length(out)
            tp,loc = istrue(out[k],y[1][3:length(y[1])],check,ıou)
            push!(predictions[numsdic[out[k][5]]],(tp,out[k][6]))
            if tp
                check[loc] = 1
            end
        end
    end
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
    return apdic
end

function istrue(prediction,labels,check,ıou)
    min = ıou
    result = false
    location = length(labels) + 1
    for i in 1:length(labels)
        if prediction[5] == namesdic[labels[i][5]] && check[i] == 0 && ıoumatch(prediction[1],prediction[2],prediction[3],prediction[4],labels[i][1],labels[i][2],labels[i][3],labels[i][4]) > min
            min = ıoumatch(prediction[1],prediction[2],prediction[3],prediction[4],labels[i][1],labels[i][2],labels[i][3],labels[i][4])
            result = true
            location = i
        end
    end
    return result ,location
end

function displaytest(file,model; record = false)
    im, img_size, img_originalsize, padding = loadprepareimage(file,(416,416))
    im_input = Array{Float32}(undef,416,416,3,1)
    im_input[:,:,:,1] = permutedims(collect(channelview(im)),[2,3,1]);
    if gpu() >= 0 im_input = KnetArray(im_input) end
    res = model(im_input)
    a = postprocessing(res,0.3,0.3)
    for i in 1:length(a)
        drawsquare(im,a[i][1],a[i][2],a[i][3],a[i][4],padding)
        FreeTypeAbstraction.renderstring!(im, string(numsdic[a[i][5]]), face, (14,14)  ,Int32(round(a[i][2]))-padding[2],Int32(round(a[i][1]))-padding[1],halign=:hleft,valign=:vtop,bcolor=eltype(im)(1.0,1.0,1.0),fcolor=eltype(im)(0,0,0)) #use `nothing` to make bcolor transparent
    end
    p1 = padding[1]
    p2 = padding[2]
    display(im[1:end-p2,1:end-p1])
    if record save("outexample.jpg",im[1:end-p2,1:end-p1]) end
end
