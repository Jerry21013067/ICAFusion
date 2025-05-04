function des = denseLSS(img,desc_rad,nrad,nang);
% denseLSS - 计算输入图像的密集局部对称结构（LSS）描述符
%
% 输入参数：
%   img - 输入图像，灰度图像或彩色图像的第一通道
%   desc_rad - 描述符的半径
%   nrad - 半径方向上的采样点数
%   nang - 角度方向上的采样点数
%
% 输出参数：
%   des - 计算得到的描述符，大小为 (des_num, des_height, des_width)
%         其中 des_num = nrad * nang

% 初始化参数
parms.patch_size=3; % 用于计算描述符的局部块大小
parms.desc_rad=desc_rad; % 描述符的半径
parms.nrad=nrad; % 半径方向上的采样点数
parms.nang=nang; % 角度方向上的采样点数
parms.var_noise=3000; % 噪声方差
parms.saliency_thresh = 1; % 显著性阈值
%parms.saliency_thresh = 0.7; % 可选的显著性阈值
parms.homogeneity_thresh=1; % 均质性阈值
%parms.homogeneity_thresh=0.7; % 可选的均质性阈值
parms.snn_thresh=1; % 最近邻阈值，通常禁用显著性检查
%parms.snn_thresh=0.85; % 可选的最近邻阈值
%parms.nChannels=size(i,3); % 图像的通道数
des_num = parms.nrad*parms.nang; % 描述符的维度
%des_num = (2*desc_rad+1)*(2*desc_rad+1); % 可选的描述符维度计算方式
margin = parms.desc_rad + (parms.patch_size-1)/2; % 计算需要填充的边界大小
img = padarray(img,[margin,margin],'symmetric'); % 对图像进行边界填充
img = double(img(:,:,1)); % 将图像转换为双精度类型，并取第一通道
[h,w] = size(img); % 获取图像的高度和宽度


des_width = w-2*margin; % 描述符的宽度
des_height = h-2*margin; % 描述符的高度

destmp = zeros(des_height,des_width,des_num); % 初始化临时描述符矩阵
des = zeros(h,w,des_num); % 初始化最终描述符矩阵

% 调用 mex 文件计算描述符
%[resp, draw_coords, salient_coords, homogeneous_coords, snn_coords] = mexCalcSsdescs(img, parms);
[resp, draw_coords, salient_coords, homogeneous_coords, snn_coords] = mexCalcSsdescs1(img, parms);
%[resp, draw_coords, salient_coords, homogeneous_coords, snn_coords] = mexCalcSsdescs_mean(img, parms);
%[resp, draw_coords, salient_coords, homogeneous_coords, snn_coords] = mexCalcSSD(img, parms);
%[resp, draw_coords, salient_coords, homogeneous_coords, snn_coords] = mexCalcSSDslow(img, parms);

% 转置响应矩阵
resp = resp';
% 重塑响应矩阵
temp = reshape(resp,[des_width,des_height,des_num]);
% 调整维度顺序
temp1 = permute(temp,[2 1 3]);

% 将临时描述符矩阵赋值给最终描述符矩阵
destmp = temp1;

% 将描述符矩阵转换为单精度类型
%des(margin:h-margin-1,margin:w-margin-1,:) = destmp;
%des(margin+1:h-margin,margin+1:w-margin,:) = destmp;
%des = single(des);
des =single(destmp); % 将描述符矩阵转换为单精度类型
des = permute(des, [3 1 2]); % 调整描述符矩阵的维度顺序

