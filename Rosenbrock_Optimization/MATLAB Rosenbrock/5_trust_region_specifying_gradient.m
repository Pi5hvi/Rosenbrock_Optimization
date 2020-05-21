options = optimoptions('fminunc','display','iter','Algorithm','trust-region','SpecifyObjectiveGradient',true);
x0=[-10,-10];
fun=@rosenbrockwithgrad;
tic
x=fminunc(fun,x0,options)
toc