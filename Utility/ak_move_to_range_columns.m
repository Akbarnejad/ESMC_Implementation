function [output] = ak_move_to_range_columns( input )
%

[N,F] = size(input);
input = ak_zero_mean(input);
output = input;
for f=1:F
   temp = input(:,f);
   temp = max(max(temp));
   output(:,f) = output(:,f)/temp;
end

end

