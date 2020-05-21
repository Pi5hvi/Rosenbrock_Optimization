%% Computation time and solving with grad from (0,0)
options = optimoptions('fminunc','display','iter','Algorithm','trust-region','SpecifyObjectiveGradient',true);
x0=[0,0];
fun=@rosenbrockwithgrad;
tic
x=fminunc(fun,x0,options);
toc
fprintf('\nThe minimum occurs at (%d,%d)\n',x)
fprintf('\nClearly, the number of itterations and function evaluations are less than using fminnunc without gradient.\nHowever, the wall clock time seems to be more. The results are far more accurate')
%% Computation time and solving with grad from (-10,-10)
x0=[-10,-10];
tic
x=fminunc(fun,x0,options);
toc
fprintf('\nThe minimum occurs at (%d,%d)\n',x)
fprintf('\nClearly, the number of itterations and function evaluations are more than using fminnunc without gradient.\nAlso, the wall clock time seems to be more. The results are far more accurate')