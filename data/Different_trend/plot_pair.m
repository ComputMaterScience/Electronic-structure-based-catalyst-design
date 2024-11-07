clear all;clc; figure; hold on;

data = load('data.txt');

ylabel_o = {'Q(*)','Q(*N)','Q(*NH)','Q(*NH_2)'};

[S,AX,BigAx,H,HAx] = plotmatrix(data);

for i = 1:size(H,2)
    H(i).NumBins = 7;
end

AX(1,1).YLabel.String = ylabel_o{1};
AX(2,1).YLabel.String = ylabel_o{2};
AX(3,1).YLabel.String = ylabel_o{3};
AX(4,1).YLabel.String = ylabel_o{4};

AX(4,1).XLabel.String = ylabel_o{1};
AX(4,2).XLabel.String = ylabel_o{2};
AX(4,3).XLabel.String = ylabel_o{3};
AX(4,4).XLabel.String = ylabel_o{4};
