clear all;clc; figure; hold on;

data = load('data.txt');

ylabel_o = {'Q(*)','Q(*N)','Q(*NH)','Q(*NH_2)'};

% Set [min,max] value of C to scale colors
% This must span the range of your data!
clrLim = [0,1];  % or [-1,1] as in your question

% Set the  [min,max] of diameter where 1 consumes entire grid square
diamLim = [0.3, 1];

imagesc(corrcoef(data))
colormap(gca,'hot');
colorbar();
caxis(clrLim);
axis equal
axis tight
view(90,90)
