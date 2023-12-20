clear all;
close all;
clc;
addpath('./Utils');

% PatSize 

PatSize = 3;
% n=3;

fprintf(' ... ... read image file ... ... ... ....\n');
im1 = imread('./bern/sar/1.bmp');
im2 = imread('./bern/sar/2.bmp');
im_gt = imread('./bern/sar/gt.bmp');
im_bw = imread('./bern/BW.bmp');
im_bw = double(im_bw); %%二值显著图

fprintf(' ... ... read image file finished !!! !!!\n\n');

im1 = double(im1(:,:,1));
im2 = double(im2(:,:,1));
im_gt = double(im_gt(:,:,1));
im_gt = uint8(im_gt/255);

[ylen, xlen] = size(im1);
GT_lab = reshape(im_gt,ylen*xlen,1);


nrmap = imread('./bern/DI2.bmp');
% imshow(nrmap);
nrmap = im2double(nrmap(:,:,1)); %%显著性差异图
imshow(nrmap);

%对图像进行聚类分析
feat_vec = reshape(nrmap, ylen*xlen, 1);
im_lab = gao_clustering(feat_vec, ylen, xlen);
% imshow(im_lab);
save(['./bern/','fcm2.mat'],'im_lab')

% im_lab = importdata('./bern/fcm2.mat');
% imshow(im_lab); %%聚类标签

% all samples
%生成样本：nrmap(50601的元胞，每个元胞代表一个patsize*patsieze的样本)

mag = (PatSize-1)/2;
imTmp = zeros(ylen+PatSize-1, xlen+PatSize-1);
imTmp((mag+1):end-mag,(mag+1):end-mag) = nrmap; 

nrmap = im2col_general(imTmp, [PatSize, PatSize]);
nrmap = mat2imgcell(nrmap, PatSize, PatSize, 'gray');

%%非显著性区域
idx = find(im_bw == 255);
ex_idx = find(im_bw == 0);
ex_num = numel(ex_idx);
ex_lab = zeros(ex_num,1);
ex_data = [ex_idx,ex_lab];
save(['./bern/sample/','ex_data.mat'], 'ex_data')

%%显著性的区域
pos_idx = find(im_lab == 0 & im_bw == 255);
neg_idx = find(im_lab == 1 & im_bw == 255);
tst_idx = find(im_lab == 0.5 & im_bw == 255);

%%总的可以用于训练的样本
PosNum = numel(pos_idx);
NegNum = numel(neg_idx);
TstNum = numel(tst_idx);
% Index and label of training samples

pos_lab = ones(PosNum,1);
neg_lab = zeros(NegNum,1);
train_lab = [neg_lab;pos_lab];
train_idx = [neg_idx;pos_idx];
train_data = [train_idx,train_lab];

save(['./bern/sample/','train_data.mat'], 'train_data')

% 从训练样本中选取部分用于训练
pos_idx1 = pos_idx(randperm(numel(pos_idx)));%%打乱索引顺序
neg_idx1 = neg_idx(randperm(numel(neg_idx)));
% % %%样本个数视具体数据集定
PosNum1 = 3400;
NegNum1 = 6000;

% Index and label of training samples

pos_lab1 = ones(PosNum1,1);
neg_lab1 = zeros(NegNum1,1);
tra_pos_idx = pos_idx1(1:PosNum1,:);
tra_neg_idx = neg_idx1(1:NegNum1,:);
train_lab1 = [neg_lab1;pos_lab1];
train_idx1 = [tra_neg_idx;tra_pos_idx];
train_data1 = [train_idx1,train_lab1];

pos1 = [tra_pos_idx,pos_lab1];
neg1 = [tra_neg_idx,neg_lab1];
save(['./bern/sample/','pos1.mat'], 'pos1')
save(['./bern/sample/','neg1.mat'], 'neg1')


fid = fopen('./bern/sample/train1.txt','wt');
for i=1:PosNum1+NegNum1
    idx_i = train_data1(i,1);
    lab_i = train_data1(i,2);
        fprintf(fid,'%g',lab_i);
        fprintf(fid,'%s','/img_');
        fprintf(fid,'%g',idx_i);
        fprintf(fid,'%s\t','.png');
        fprintf(fid,'%g\n',lab_i);

end

fclose(fid);


% Training samples generation

  for i = 1:NegNum1
       pic = nrmap{tra_neg_idx(i)};
       imwrite(pic,['./bern/sample/train1/0/','img_',num2str(tra_neg_idx(i)),'.png'],'png');

   end

  for i = 1:PosNum1
      pic = nrmap{tra_pos_idx(i)};
      imwrite(pic,['./bern/sample/train1/1/','img_',num2str(tra_pos_idx(i)),'.png'],'png');
  end

save(['./bern/sample/','train_data1.mat'], 'train_data1') 
% %train_Ottawa.mat train_data3 = [train_idx,train_lab]训练本的索引和标签

fprintf(' ... .. over ..\n');




