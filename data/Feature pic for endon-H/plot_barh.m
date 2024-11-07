clear all; clc; figure; hold on;

% fid = fopen('lasso.txt','r');
% fid = fopen('forest.txt','r');
% fid = fopen('permu.txt','r');
fid = fopen('shap.txt','r');
lasso_name = cell(1);
lasso_data = zeros(1,1);
cnt = 0;
while ~feof(fid)
    line = split(fgetl(fid));
    cnt = cnt + 1;
    lasso_name{cnt} = string(line(1));
    lasso_data(cnt) = str2double(line(2));
end
fclose(fid);

barh(1:size(lasso_name,2),lasso_data);
yticks(1:size(lasso_name,2))
yticklabels(lasso_name)

xlabel('Importance')

box on;
axis square;
set(gca,'LineWidth',1.5);
set(gca,'FontSize',14);