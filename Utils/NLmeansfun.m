%NL-means�㷨��ȥ��Ч��������ȱ��������ʱ��̫��
function DenoisedImg=NLmeansfun(I,ds,Ds,h)
%I:������ͼ��
%ds:���򴰿ڰ뾶
%Ds:�������ڰ뾶
%h:��˹����ƽ������
%DenoisedImg��ȥ��ͼ��
I=double(I);
[m,n]=size(I);
DenoisedImg=zeros(m,n);         %����Ϊȫ0����
PaddedImg = padarray(I,[ds,ds],'symmetric','both');%��չ�����С��padarry�������÷�
kernel=ones(2*ds+1,2*ds+1);     %d=2*ds+1
kernel=kernel./((2*ds+1)*(2*ds+1));%��ʱkernel������ֵΪ0.04
h2=h*h;                         
for i=1:m                       %mΪ����
    for j=1:n                   %nΪ����
        i1=i+ds;                %��i1��j1��Ϊ��������
        j1=j+ds;
        W1=PaddedImg(i1-ds:i1+ds,j1-ds:j1+ds);%���򴰿�1
        wmax=0;
        average=0;
        sweight=0;
        %%��������
        rmin = max(i1-Ds,ds+1);
        rmax = min(i1+Ds,m+ds);
        smin = max(j1-Ds,ds+1);
        smax = min(j1+Ds,n+ds);
        for r=rmin:rmax
            for s=smin:smax
                if(r==i1&&s==j1)    
                continue;
                end
                W2=PaddedImg(r-ds:r+ds,s-ds:s+ds);%���򴰿�2
                Dist2=sum(sum(kernel.*(W1-W2).*(W1-W2)));%��������
                w=exp(-Dist2/h2);           % wȨֵ
                if(w>wmax)
                    wmax=w;
                end
                sweight=sweight+w;          % Ȩֵ���ۼӺ�
                average=average+w*PaddedImg(r,s);
            end
        end
        average=average+wmax*PaddedImg(i1,j1);%����ȡ���Ȩֵ
        sweight=sweight+wmax;
        DenoisedImg(i,j)=average/sweight;
    end
end