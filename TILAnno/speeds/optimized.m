function phi = optimized(Img, phi0, phi, iter_refine1, iter_refine2)
phi(phi>=0)=0;phi(phi<0)=1;
phi = imdilate(phi,strel('disk',1));
phi(phi0==1)=1;
phi = refine(Img,phi,iter_refine1,'mean');

%third iteration
phi(phi<=0)=0;phi(phi>0)=1;
phi = imerode(phi,strel('disk',2));
phi(phi0==1)=1;
phi = refine(Img,phi,iter_refine2,'mean_var');
end