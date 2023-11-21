function [w,f,normgrad] = SINewton(fun,gfun,Hvec,Y,w,bsz,kmax,tol)
rho = 0.1;
gam = 0.9;
jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
eta = 0.5;
CGimax = 10; % max number of CG iterations
n = size(Y,1);
[n,~] = size(Y);
I = 1:n;
f = zeros(kmax + 1,1);
f(1) = fun(I,w);
normgrad = zeros(kmax,1);
nfail = 0;
nfailmax = 5*ceil(n/bsz);
for k = 1 : kmax
    Ig = randperm(n,bsz);
    IH = randperm(n,bsz);
    Mvec = @(v)Hvec(IH,w,v);
    b = gfun(Ig,w);
    normgrad(k) = norm(b);
    s = CG(Mvec,-b,-b,CGimax,rho);
    a = 1;
    f0 = fun(Ig,w);
    aux = eta*b'*s;
    for j = 0 : jmax
        wtry = w + a*s;
        f1 = fun(Ig,wtry);
        if f1 < f0 + a*aux
%             fprintf('Linesearch: j = %d, f1 = %d, f0 = %d, |as| = %d\n',j,f1,f0,norm(a*s));
            break;
        else
            a = a*gam;
        end
    end
    if j < jmax
        w = wtry;
    else
        nfail = nfail + 1;
    end
    f(k + 1) = fun(I,w);
    if mod(k,100)==0
        fprintf('k = %d, a = %d, f = %d, ||g|| = %d\n',k,a,f(k+1),normgrad(k));
    end
    if nfail > nfailmax
        f(k+2:end) = [];
        normgrad(k+1:end) = [];
        fprintf('stop iteration as linesearch failed more than %d times\n',nfailmax);
        break;
    end
    if normgrad(k) < tol
        break;
    end
end
fprintf('k = %d, a = %d, f = %d, ||g|| = %d\n',k,a,f(k+1),normgrad(k));
end
        
        
    
    
