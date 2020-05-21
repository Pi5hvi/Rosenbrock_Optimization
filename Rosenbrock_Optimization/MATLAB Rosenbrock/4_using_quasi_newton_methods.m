a=1;
b=100;
fun =@(x)(a-x(1)).^2+b.*(x(2)-x(1).^2).^2;
options = optimoptions('fminunc','display','iter','Algorithm','quasi-newton', 'FiniteDifferenceStepSize',1e-12,'MaxIterations',400000,'SpecifyObjectiveGradient',false);
%Setting display itteration, as well as changing algorithm to quasi-newton
%as for some reason it was taking trust-region algorithm and throwing up a
%warning that gradient needs to be provided before switching over to
%quasi-newton algorithm.
x0=[-10,-10];
tic
[x,fval] = fminunc(fun,x0,options);
toc