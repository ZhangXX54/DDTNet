function phi = refine(img,mask,iterations,speed_functional)
%   iterations = 200;

%   paths;
%   clf; colormap gray;

  %- load image for truck, let user select point
%   load('images/001rgb');

%   img = double(rgb2gray(imread(['images/00',num2str(i),'.png'])));
  se1=strel('disk',1);
  mask = logical(imdilate(mask,se1));
  %disp('draw initial segmentation: left-click marks boundary points, right-click finishes');
  %mask = get_blob_mask(img);
  
  %- speed functional
  if strcmp(speed_functional,'mean')
      h = mean_speed();
  elseif strcmp(speed_functional,'mean_var')
      h = mean_var_speed();
  elseif strcmp(speed_functional,'georgiou')
      h = georgiou_speed();
  elseif strcmp(speed_functional,'bhattacharyya')
      h = bhattacharyya_speed();
  else 
      h = threshold_speed();
  end
  %- initialize
  [phi C] = mask2phi(mask);
  h.init(img, phi, C); % initialize statistics
  
  %- curve evolution
  [phi_ C_] = ls_sparse(phi, C, h, iterations);

  %- display results
%   clf; imagesc(img); axis image off;
%   hold on;
% %   contour(phi,  [0 0], 'g', 'LineWidth', 1); % initial
% %   contour(phi_, [0 0], 'r', 'LineWidth', 1); % final
%   contour(phi,  [0 0], 'g'); % initial
%   contour(phi_, [0 0], 'r'); % final
%   hold off;
%   
  phi = phi_;
%   imwrite(phi_,'result.png')
%  saveas(gcf,['MASK_',num2str(i),'_mean_var.png']);
end

