function y=circonv(x1,x2,N)

nx1=0:length(x1)-1;
nx2=0:length(x2)-1;
x_1=[x1 zeros(1,N-length(x1))];
h_1=[x2 zeros(1,N-length(x2))];
y1=conv(x_1,h_1);
z_1=[zeros(1,N) y1(1:(N-1))];
z_2=[y1((N+1):(2*N-1)) zeros(1,N)];
z=z_1(1:(2*N-1))+z_2(1:(2*N-1))+y1(1:(2*N-1));
y=z(1:N);
ny=0:N-1;

end