clear all;
close all;
clc;
addpath('./Utils');

% PatSize 

PatSize = 3;
% n=5;

fprintf(' ... ... read image file ... ... ... ....\n');
im1 = imread('./river/sar/1.bmp');
im2 = imread('./river/sar/2.bmp');
im_gt = imread('./river/sar/gt.bmp');
im_bw = imread('./river/BW2.bmp');
im_bw = double(im_bw); %%二值显著图

fprintf(' ... ... read image file finished !!! !!!\n\n');

im1 = double(im1(:,:,1));
im2 = double(im2(:,:,1));
im_gt = double(im_gt(:,:,1));
im_gt = uint8(im_gt/255);

[ylen, xlen] = size(im1);
GT_lab = reshape(im_gt,ylen*xlen,1);

% Caculate neighborhood-based ratio image
fprintf(' ... .. compute the neighborhood ratio ..\n');

nrmap = imread('./river/DI2.bmp');
nrmap = im2double(nrmap(:,:,1));% 加载稀疏表示之后的图像

im_lab = importdata('./river/fcm2.mat');
% imshow(im_lab);
% Label of preclassification

tst_idx = find(im_lab == 0.5 & im_bw == 255);
tst_idx=sortrows(tst_idx);

% Select patch for each pixel center

mag = (PatSize-1)/2;
imTmp = zeros(ylen+PatSize-1, xlen+PatSize-1);
imTmp((mag+1):end-mag,(mag+1):end-mag) = nrmap; 

nrmap = im2col_general(imTmp, [PatSize, PatSize]);%%5*5，50601
nrmap = mat2imgcell(nrmap, PatSize, PatSize, 'gray');%转换成50601*1的元胞

TstNum = numel(tst_idx);


%测试数据的标签取自参考图
tst_lab = zeros(TstNum,1); 
for i=1:ylen*xlen
    for s=1:TstNum
        if tst_idx(s,:)==i
           tst_lab(s,:)=GT_lab(i,:);
        end
    end
end
    
tst_data = [tst_idx,tst_lab];


posNum = 0;
negNum = 0;

for i=1:TstNum
    if tst_data(i,2)==1
        posNum=posNum+1;
    else
        negNum =negNum+1;
    end
    
end

tst_poslab=ones(posNum,1);
tst_neglab=zeros(negNum,1);

tst_posidx=ones(posNum,1);
tst_negidx=ones(negNum,1);


s=1;
for i=1:TstNum
    if tst_data(i,2)==1
       tst_posidx(s,1)=tst_data(i,1);
       s=s+1;
    end
end

v=1;
for i=1:TstNum
    if tst_data(i,2)==0
       tst_negidx(v,1)=tst_data(i,1);
       v=v+1;
    end
end  

tst_posdata=[tst_posidx,tst_poslab];
tst_negdata=[tst_negidx,tst_neglab];
%%%横向拼接用逗号，纵向拼接用分号
tst_data1 = [tst_negdata;tst_posdata];
save(['./river/sample/','tst_pos.mat'], 'tst_posdata')
save(['./river/sample/','tst_neg.mat'], 'tst_negdata')

save(['./river/sample/','tst_data1.mat'], 'tst_data1')

% for i = 1:posNum
%        pic = nrmap{tst_posidx(i)};
%        imwrite(pic,['./river/sample/test1/1/','img_',num2str(tst_posidx(i)),'.png'],'png');
% 
% end
 
for i = 1:negNum
       pic = nrmap{tst_negidx(i)};
       imwrite(pic,['./river/sample/test1/0/','img_',num2str(tst_negidx(i)),'.png'],'png');

end
