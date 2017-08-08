function [auc] = ak_general_get_area_under_plot(x,y)
%
%inputs:  
%   x: a vector of size 1xN.
%   y: a vector of size 1xN.
%outputs:
%   auc: aread under the plot(x,y).


%get some constants
N = numel(x);

%compute auc
auc = 0;
for n=1:(N-1)
    x_diff = x(n+1)-x(n);
    x_diff = abs(x_diff);
    y_sum  = y(n)+y(n+1);
    to_add = 0.5*x_diff*y_sum;
    auc = auc + to_add;
end

end

