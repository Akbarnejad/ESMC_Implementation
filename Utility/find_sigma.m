function [sigma] = find_sigma(dataset)
%returns proper sigma of dataset


if(strcmp(dataset,'CAL500'))
    sigma = 18.89;
elseif(strcmp(dataset,'mediamill'))
    sigma = 1.86;
elseif(strcmp(dataset,'delicious'))
    sigma = 9.89;
elseif(strcmp(dataset,'corel5k'))
    sigma = 7.97;
elseif(strcmp(dataset,'bibtex'))
    sigma = 20.65;
elseif(strcmp(dataset,'ak_lear_DenseHue__corel5k'))
    sigma = 1230.51;
elseif(strcmp(dataset,'ak_lear_DenseHue__espgame'))
    sigma = 356.22;
elseif(strcmp(dataset,'ak_lear_DenseHue__iaprtc12'))
    sigma = 1878.13;
elseif(strcmp(dataset,'ak_lear_DenseHue__mirflickr'))
    sigma = 356.94;   
else
   error(['Invalid dataset: ' dataset]); 
end

end

