%% Create Rosenbrock function with grad
function [f,g] = rosenbrockwithgrad(x)
a=1;
b=100;
% Calculate objective f
f = (a-x(1)).^2+b.*(x(2)-x(1).^2).^2;

if nargout > 1 % gradient required
    g = [-4*b*(x(2)-x(1)^2)*x(1)-2*(a-x(1));
        2*b*(x(2)-x(1)^2)];
end