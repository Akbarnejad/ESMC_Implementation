function [ output ] = get_cross_radialbasis_kernel(x,y,sigma)
%%computes cross kernel between rows of x and rows of y
%disp('computing corss radial-basis kernel........');
%tic;

[nx,~] = size(x);
[ny,~] = size(y);

output = zeros(nx,ny);
for i=1:nx
    if(mod(i,100)==1)
       %disp(['please wait..... ' num2str(i) ' of ' num2str(nx)]); 
    end
    for j=1:ny
       temp = x(i,:)-y(j,:);
       temp = sum(sum(temp.*temp));
       output(i,j) = temp;
    end
end
%now, output contains ||xi-yj||^2 s

output = -output/(2*sigma*sigma);
output = exp(output);
%toc;
end

