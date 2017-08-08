function [C] = get_best_C(dataset)
%

if(strcmp(dataset,'CAL500'))
    C = 1000.0;
elseif(strcmp(dataset,'mediamill'))
    C = -1;
elseif(strcmp(dataset,'delicious'))
    C = -1;
elseif(strcmp(dataset,'corel5k'))
    C = 100.0;
elseif(strcmp(dataset,'bibtex'))
    C = -1;
elseif(strcmp(dataset,'ak_lear_DenseHue__corel5k'))
    C = 100.0;
elseif(strcmp(dataset,'ak_lear_DenseHue__espgame'))
    C = 0.001;
elseif(strcmp(dataset,'ak_lear_DenseHue__iaprtc12'))
    C = 0.0001;
else
   error(['Invalid dataset: ' dataset]); 
end

end

