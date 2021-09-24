%Alignment for DDTNet
clc;clear all;close all;
results_path = 'XX\';
results = importdata([results_path,'results\test.txt']);
output_path = 'XX\';
data_names = results.textdata;
data_boundingbox = results.data;
pos = [];
Pos = [];

for i = 1:length(data_names)
    pos = [pos;data_boundingbox(i,:)];

    if i == length(data_names) || ~strcmp(data_names{i},data_names{i+1}) 
        Name = strsplit(data_names{i},'/');                               %data_names{i}: ../prediction/test/Dataset Name/data1/076.png
        j = 6;
        Premask = imread([results_path,'premask_',Name{j}]);Premask = imresize(Premask,2);
        Precontour = imread([results_path,'precontour_',Name{j}]);Precontour = imresize(Precontour,2);
        Precontour = bwmorph(Precontour, 'thin', Inf);Premask(Precontour) = 0;

        pre = zeros(size(Premask));pre0 = zeros(size(Premask));
        for x = 1:size(pos,1)
             x1 = round(pos(x,3));x2 = round(pos(x,5));
             y1 = round(pos(x,2));y2 = round(pos(x,4));
             if x1 == 0
                 x1 = 1;
             end
             if y1 == 0
                 y1 = 1;
             end
             [mask,num] = bwlabel(Premask(y1:y2,x1:x2),4);
             if num > 1
                 area = [];
                 pre(y1:y2,x1:x2) = 0;
                 STATS = regionprops(mask,'Area');
                 area = cat(1,STATS.Area);
                 [maxVal, maxInd] = max(area);
                 mask(mask~=maxInd) = 0;
                 pre(y1:y2,x1:x2) = mask;
                 Pos = [Pos;cellstr(Name{j}),cellstr(num2str(x1)),cellstr(num2str(y1)),cellstr(num2str(x2)),cellstr(num2str(y2))];
             elseif num == 1
                 pre(y1:y2,x1:x2) = Premask(y1:y2,x1:x2);
%                  Pos = [Pos;x1,y1,x2,y2];
                 Pos = [Pos;cellstr(Name{j}),cellstr(num2str(x1)),cellstr(num2str(y1)),cellstr(num2str(x2)),cellstr(num2str(y2))];
             end
             Premask(pre~=0) = 0;
             pre0(pre~=0) = x;
        end
        
        se1=strel('disk',1);pre=imdilate(pre0,se1);
        pre = cat(3,uint8(pre),uint8(zeros(size(Premask))),uint8(zeros(size(Premask))));
        imwrite(pre,[output_path,'realmask_',Name{j}]);
        pos = [];
    end
end
xlswrite([output_path,'BoundingBoxes.xlsx'],Pos);