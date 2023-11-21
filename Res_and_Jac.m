%%
function [r,J] = Res_and_Jac(X,y,w)
% vector of residuals
aux = exp(-myquadratic(X,y,w));
r = log(1 + aux);
% the Jacobian matrix
a = -aux./(1+aux);
[n,d] = size(X);
d2 = d^2;
ya = y.*a;
qterm = zeros(n,d2);
for k = 1 : n
    xk = X(k,:); % row vector x
    xx = xk'*xk;
    qterm(k,:) = xx(:)';
end
Y = [qterm,X,ones(n,1)];
J = (ya*ones(1,d2+d+1)).*Y;
end
%%
function q = myquadratic(X,y,w)
d = size(X,2);
d2 = d^2;
W = reshape(w(1:d2),[d,d]);
v = w(d2+1:d2+d);
b = w(end);
qterm = diag(X*W*X');
q = y.*qterm + ((y*ones(1,d)).*X)*v + y*b;
end
