clear all;
clc;
close all;
% load final result and the training sample index
im_gt = imread('./bern/sar/gt.bmp');
im_gt = double(im_gt(:,:,1));

im_gt(im_gt==255)=1;
% im_gt1 = im_gt';
[ylen,xlen] = size(im_gt);
im_gt_1 = reshape(im_gt,ylen*xlen,1);

load('./bern/sample/train_data.mat');
load('./bern/sample/tst_data1.mat');
load('./bern/sample/ex_data.mat');
pred=importdata('./bern/sample/result_mat/result1.mat');
test_lab=uint8(pred)';
tst_data=tst_data1;
tst_data(:,2)=test_lab;
data=[train_data;tst_data;ex_data];
% data=[tst_data;ex_data];

gt_map=sortrows(data);
result=gt_map(:,2);

 aa = find(im_gt_1==0&result~=0);%%FP
 bb = find(im_gt_1~=0&result==0);%%FN
 cc = find(im_gt_1==0&result==0);%%TN
 dd = find(im_gt_1~=0&result~=0);%%TP
 
 FP = numel(aa);
 FN = numel(bb);
 TN = numel(cc);
 TP = numel(dd);
 
 M=xlen*ylen;
 OE = FP + FN; 
 PCC = 1-OE/M;  
 B=(TP/M+FP/M)*(TP/M+FN/M)+(TN/M+FN/M)*(TN/M+FP/M);
 kappa=(PCC-B)/(1-B);
 
 CDM=reshape(result,ylen,xlen);
 imshow(CDM,[]);
 imwrite(CDM,'./bern/sample/result_cdm/1.bmp');
 
fid = fopen('./bern/sample/result_cdm/1.txt', 'a');
fprintf(fid, 'FN : %d \n', FN);
fprintf(fid, 'FP : %d \n', FP);
fprintf(fid, 'OVERALL ERROR: %d \n', OE);
fprintf(fid, 'PCC          : %f \n\n\n', PCC);
fprintf(fid, 'kappa        : %f \n\n\n', kappa);
fclose(fid);

fprintf('FN : %d \n', FN);
fprintf('FP : %d \n', FP);
fprintf('OVERALL ERROR: %d \n', OE);
fprintf('PCC          : %f \n', PCC);
fprintf('kappa        : %f \n\n\n', kappa);






