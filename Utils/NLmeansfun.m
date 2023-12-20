%NL-means算法，去斑效果显著，缺点是消耗时间太长
function DenoisedImg=NLmeansfun(I,ds,Ds,h)
%I:含噪声图像
%ds:邻域窗口半径
%Ds:搜索窗口半径
%h:高斯函数平滑参数
%DenoisedImg：去噪图像
I=double(I);
[m,n]=size(I);
DenoisedImg=zeros(m,n);         %先设为全0矩阵
PaddedImg = padarray(I,[ds,ds],'symmetric','both');%扩展矩阵大小，padarry函数的用法
kernel=ones(2*ds+1,2*ds+1);     %d=2*ds+1
kernel=kernel./((2*ds+1)*(2*ds+1));%此时kernel的像素值为0.04
h2=h*h;                         
for i=1:m                       %m为行数
    for j=1:n                   %n为列数
        i1=i+ds;                %（i1，j1）为邻域中心
        j1=j+ds;
        W1=PaddedImg(i1-ds:i1+ds,j1-ds:j1+ds);%邻域窗口1
        wmax=0;
        average=0;
        sweight=0;
        %%搜索窗口
        rmin = max(i1-Ds,ds+1);
        rmax = min(i1+Ds,m+ds);
        smin = max(j1-Ds,ds+1);
        smax = min(j1+Ds,n+ds);
        for r=rmin:rmax
            for s=smin:smax
                if(r==i1&&s==j1)    
                continue;
                end
                W2=PaddedImg(r-ds:r+ds,s-ds:s+ds);%邻域窗口2
                Dist2=sum(sum(kernel.*(W1-W2).*(W1-W2)));%邻域间距离
                w=exp(-Dist2/h2);           % w权值
                if(w>wmax)
                    wmax=w;
                end
                sweight=sweight+w;          % 权值的累加和
                average=average+w*PaddedImg(r,s);
            end
        end
        average=average+wmax*PaddedImg(i1,j1);%自身取最大权值
        sweight=sweight+wmax;
        DenoisedImg(i,j)=average/sweight;
    end
end