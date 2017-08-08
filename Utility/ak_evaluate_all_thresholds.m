

%SETTINGS
STEP_SIZE = 0.01;


%all_candidates = [min(min(y_hat)): STEP_SIZE : max(max(y_hat))];
all_candidates = [-1:STEP_SIZE:1];
all_accuracy = zeros(1,numel(all_candidates));
all_micro_f1 = zeros(1,numel(all_candidates));
all_macro_f1 = zeros(1,numel(all_candidates));

computed_y = y_hat;
for counter=1:numel(all_candidates)
   
   if(mod(counter,10)==1)
    disp(['iterations: ' num2str(counter) ' of ' num2str(numel(all_candidates))]);
   end
   
   
   t = all_candidates(counter);
   computed_y(:) = 0;
   computed_y(y_hat>t) = 1;
   [macro_F1, micro_F1, accuracy] = calc_measure(Y_test, computed_y);
   all_accuracy(1,counter) = accuracy;
   all_micro_f1(1,counter) = micro_F1;
   all_macro_f1(1,counter) = macro_F1;
end

subplot(3,1,1),plot(all_candidates,all_accuracy),title('accuracy');
subplot(3,1,2),plot(all_candidates,all_micro_f1),title('micro-f1');
subplot(3,1,3),plot(all_candidates,all_macro_f1),title('macro-f1');
