clear all
clc
%set values of a and b
a=1;
b=100;
%% Define function as a one-line anonymous function
fun =@(x)(a-x(1)).^2+b.*(x(2)-x(1).^2).^2;
%% Use optimotions
options = optimoptions('fminunc','display','iter','Algorithm','quasi-newton','SpecifyObjectiveGradient',false);
%Setting display itteration, as well as changing algorithm to quasi-newton as for some reason it was taking 
%trust-region algorithm and throwing up a warning that gradient needs to be provided before switching over to
%quasi-newton algorithm.
%% Use fminunc from (0,0)
x0=[0,0];
tic
[x,fval] = fminunc(fun,x0,options);
toc
fprintf('\nThe minimum occurs at (%d,%d)\n',x)
% Note tic toc is added to compute time
%% Use fminunc from (-10,-10)
x0=[-10,-10];
[x1,fval1] = fminunc(fun,x0,options);
fprintf('The minimum occurs at (%d,%d)\n',x1)
%% Fix fminunc from (-10,-10)
fprintf('The error in the calculation occurs due to the large step size taken by the function. Reducing step size,\n')
options = optimoptions('fminunc','display','iter','Algorithm','quasi-newton', 'FiniteDifferenceStepSize',1e-12,'MaxIterations',400000,'SpecifyObjectiveGradient',false);
tic
[x2,fval2] = fminunc(fun,x0,options);
toc
fprintf('\nThe minimum occurs at (%d,%d)\nA more accurate result was obtained by reducing the step size',x2)
