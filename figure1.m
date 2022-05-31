%%
t = linspace(0,2*pi,500);
y = 100*sin(t);
rand()
noise = 15*rand(1,500);
y = y + noise;
figure;
plot(t,y);
xlabel('t');
ylabel('sin(t)+noise')
%%
yy1 = smooth(y,30);
figure;
plot(t,y,'k:');
hold on
plot(t,yy1,'k','LineWidth',3);
xlabel('t');
ylabel('moving');
legend('加噪波形','平滑后')
