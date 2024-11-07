clear all;clc; figure; hold on;

data = load('data2.txt');

ylabel_o = {'Q(*)','Q(*N)','Q(*NH)','Q(*NH_2)'};

datax = 1;
datay = 2;

a = 0.307;
b = 0.64;

ida = find(data(:,8) == 0);
ids = find(data(:,8) == 1);

h = scatter(data(ida,datax),data(ida,datay),'b','filled');
h.MarkerFaceColor = '#11468F';
h.SizeData = 50;

h = scatter(data(ids,datax),data(ids,datay),'b','filled');
h.MarkerFaceColor = '#DA1212';
h.Marker = 'square';
h.SizeData = 70;

x_min = min(data(ida,datax));
x_max = max(data(ida,datax));

x = x_min:0.01:x_max;
y = a*x+b;
plot(x,y,'k-','LineWidth',2.5);

l = legend('TM-ads','TM-S','Location','southeast');
l.LineWidth = 0.5;

xlabel(ylabel_o{datax});
ylabel(ylabel_o{datay});

box on;
axis square;
set(gca,'LineWidth',1.5);
set(gca,'FontSize',14);