function initial = Seed2Contour(Seed, D)

initial = zeros(size(Seed));
[mask,~] = bwlabel(Seed,8);
STATS = regionprops(mask,'Centroid'); Centroid = cat(1,STATS.Centroid);
c = round(Centroid(:,1));r = round(Centroid(:,2));

d1 = floor(D/2);d2 = floor(D/4);
for i = 1:length(Centroid)
    gt1 = zeros(size(mask));
    gt1(r(i),c(i)) = 1;
    if length(r) == 1
        gt1 = imdilate(gt1,strel('disk',d1));
        gt1 = imerode(gt1,strel('disk',1));
    else
        A = ones(length(r)-1,1) * [r(i),c(i)] ;
        if i == 1
            B = [r(2:end),c(2:end)];
        else
            B = [r(1:i-1),c(1:i-1);r(i+1:end),c(i+1:end)];
        end
        d = sqrt(sum((B-A).^2,2));
        if min(d) >= D
            gt1 = imdilate(gt1,strel('disk',d1));
            gt1 = imerode(gt1,strel('disk',1));
        else  
            gt1 = imdilate(gt1,strel('disk',d2)); 
        end
    end
    initial(gt1==1)=1;
end
end
