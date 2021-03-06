%  This Matlab code demonstrates a high-quality lymphocyte mask generation method (TILAnno) in the following paper:
%
%  Xiaoxuan Zhang, Xiongfeng Zhu, Kai Tang, Yinghua Zhao, Zixiao Lu, Qianjin Feng, "DDTNet: A dense dual-task network 
%  for tumor-infiltrating lymphocyte detection and segmentation in histopathological images of breast cancer",
%  Medical Image Analysis 78, 2022,
clear all;
close all;
clc

%Add Path
addpath('speeds');addpath('util')
Imgpath = 'example/Img/';
Seedpath = 'example/Seed/';
Imgname = dir([Imgpath,'*.tif']);

%Save Path
Savepath = 'example/Results/';

for i = 1:length(Imgname)
%% data loading
img = imread([Imgpath, Imgname(i).name]); % real miscroscope image of cells
Img = double(rgb2gray(img));

Seed = imread([Seedpath, Imgname(i).name]); 

%% parameter setting
timestep=5;  % time step
mu=0.2/timestep;  % coefficient of the distance regularization term R(phi)
iter_inner=5;
iter_outer=40;
iter_refine1 = 100; 
iter_refine2 = 150; 
lambda=5; % coefficient of the weighted length term L(phi)
alfa=1.5;  % coefficient of the weighted area term A(phi)
epsilon=1.5; % papramater that specifies the width of the DiracDelta function

sigma=1.5;     % scale parameter in Gaussian kernel
G=fspecial('gaussian',15,sigma);
Img_smooth=conv2(Img,G,'same');  % smooth image by Gaussiin convolution
[Ix,Iy]=gradient(Img_smooth);
f=Ix.^2+Iy.^2;
g=1./(1+f);  % edge indicator function.

%% Initial contour
initial = Seed2Contour(Seed, 15);
% initialize as binary step function
c0=2;
phi0 = double(initial);
initial(initial~=0)=-c0;
phi=initial;

%% start level set evolution 
%first iteration
potential=2;  
if potential ==1
    potentialFunction = 'single-well';  % use single well potential p1(s)=0.5*(s-1)^2, which is good for region-based model 
elseif potential == 2
    potentialFunction = 'double-well';  % use double-well potential in Eq. (16), which is good for both edge and region based models
else
    potentialFunction = 'double-well';  % default choice of potential function
end
for n=1:iter_outer
    phi = drlse_edge(phi, g, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);
    if mod(n,2)==0
        figure(2);
        imagesc(Img,[0, 255]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
    end
end

%second & third iteration
phi = optimized(Img, phi0, phi, iter_refine1, iter_refine2);

% phi(phi<=0)=0;phi(phi>0)=1;
% phi = imerode(phi,strel('disk',1));
% phi2 = imerode(phi0,strel('disk',0));phi(phi2==1)=1;
% phi = refine(Img,phi,80,'mean');
 

figure(2);
imagesc(img,[0, 255]); axis off; axis equal;hold on;  
contour(phi, [0,0], 'r');hold on;  
contour(initial, [0,0], 'g'); 
str= 'Initial & TILAnno Contour ';
title(str);

phi(phi<=0)=0;phi(phi>0)=1;
phi =  bwareaopen(phi,10);
imwrite(phi,[Savepath, 'TILAnno_',Imgname(i).name]);
saveas(gcf,[Savepath, 'CV_', Imgname(i).name]);
end




