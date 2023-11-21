function mnist_2categories_quadratic_NLLS_Nest()
close all
fsz = 20;
mdata = load('mnist.mat');
imgs_train = mdata.imgs_train;
imgs_test = mdata.imgs_test;
labels_test = mdata.labels_test;
labels_train = mdata.labels_train;
%% find 1 and 7 in training data
ind1 = find(double(labels_train)==2);
ind2 = find(double(labels_train)==8);
n1train = length(ind1);
n2train = length(ind2);
fprintf("There are %d 1's and %d 7's in training data\n",n1train,n2train);
train1 = imgs_train(:,:,ind1);
train2 = imgs_train(:,:,ind2);
%% find 1 and 7 in test data
itest1 = find(double(labels_test)==2);
itest2 = find(double(labels_test)==8);
n1test = length(itest1);
n2test = length(itest2);
fprintf("There are %d 1's and %d 7's in test data\n",n1test,n2test);
test1 = imgs_test(:,:,itest1);
test2 = imgs_test(:,:,itest2);
%% plot some data from category 1
figure; colormap gray
for j = 1:20
    subplot(4,5,j);
    imagesc(train1(:,:,j));
end
%% plot some data from category 2
figure; colormap gray
for j = 1:20
    subplot(4,5,j);
    imagesc(train2(:,:,j));
end
%% use PCA to reduce dimensionality of the problem to 20
[d1,d2,~] = size(train1);
X1 = zeros(n1train,d1*d2);
X2 = zeros(n2train,d1*d2);
for j = 1 : n1train
    aux = train1(:,:,j);
    X1(j,:) = aux(:)';
end
for j = 1 :n2train
    aux = train2(:,:,j);
    X2(j,:) = aux(:)';
end
X = [X1;X2];
D1 = 1:n1train;
D2 = n1train+1:n1train+n2train;
[U,Sigma,~] = svd(X','econ');
esort = diag(Sigma);
figure;
plot(esort,'.','Markersize',20);
grid;
nPCA = 20;
Xpca = X*U(:,1:nPCA); % features
figPCA = figure; 
hold on; grid;
plot3(Xpca(D1,1),Xpca(D1,2),Xpca(D1,3),'.','Markersize',20,'color','k');
plot3(Xpca(D2,1),Xpca(D2,2),Xpca(D2,3),'.','Markersize',20,'color','r');
view(3)
%% split the data to training set and test set
Xtrain = Xpca;
Ntrain = n1train + n2train;
Xtest1 = zeros(n1test,d1*d2);
Xtest = zeros(n2test,d1*d2);
for j = 1 : n1test
    aux = test1(:,:,j);
    Xtest1(j,:) = aux(:)';
end
for j = 1 :n2test
    aux = test2(:,:,j);
    Xtest2(j,:) = aux(:)';
end
Xtest = [Xtest1;Xtest2]*U(:,1:nPCA);
Ntest = n1test+n2test;
testlabel = ones(Ntest,1);
testlabel(n1test+1:Ntest) = -1;
%% category 1 (1): label 1; category 2 (7): label -1
label = ones(Ntrain,1);
label(n1train+1:Ntrain) = -1;
%% dividing quadratic surface
%% optimize w and b 
d = nPCA;
w = ones(d^2+d+1,1);
% params for SINewton
bsz = ceil(Ntrain/10);
no_epochs = 5e2;
tol = 1e-3;
batch_size=bsz;
%
[w,f,gnorm] = SGDNest(Xtrain,label,w,batch_size,no_epochs,tol);
figure;
plot(f,'Linewidth',2);
xlabel('iter','fontsize',fsz);
ylabel('f','fontsize',fsz);
set(gca,'fontsize',fsz,'Yscale','log');
figure;
plot(gnorm,'Linewidth',2);
xlabel('iter','fontsize',fsz);
ylabel('||g||','fontsize',fsz);
set(gca,'fontsize',fsz,'Yscale','log');

%% apply the results to the test set
test = myquadratic(Xtest,testlabel,1:Ntest,w);
hits = find(test > 0);
misses = find(test < 0);
nhits = length(hits);
nmisses = length(misses);
fprintf('n_correct = %d, n_wrong = %d, accuracy %d percent\n',nhits,nmisses,nhits/Ntest);
%% plot the dividing surface if nPCA = 3
if d == 3
    d2 = d^2;
    W = reshape(w(1:d2),[d,d]);
    v = w(d2+1:d2+d);
    b = w(end);
    xmin = min(Xtrain(:,1)); xmax = max(Xtrain(:,1));
    ymin = min(Xtrain(:,2)); ymax = max(Xtrain(:,2));
    zmin = min(Xtrain(:,3)); zmax = max(Xtrain(:,3));
    nn = 50;
    figure(figPCA);
    [xx,yy,zz] = meshgrid(linspace(xmin,xmax,nn),linspace(ymin,ymax,nn),...
        linspace(zmin,zmax,nn));
    qsurf = W(1,1)*xx.^2+W(2,2)*yy.^2+W(3,3)*zz.^2+(W(1,2)+W(2,1))*xx.*yy...
        +(W(1,3)+W(3,1))*xx.*zz++(W(2,3)+W(3,2))*yy.*zz...
        +v(1)*xx+v(2)*yy+v(3)*zz+b;
    p = patch(isosurface(xx,yy,zz,qsurf,0));
    p.FaceColor = 'cyan';
    p.EdgeColor = 'none';
    camlight 
    lighting gouraud
    alpha(0.3);
end
end
%%
function q = myquadratic(Xtrain,label,I,w)
X = Xtrain(I,:);
d = size(X,2);
d2 = d^2;
y = label(I);
W = reshape(w(1:d2),[d,d]);
v = w(d2+1:d2+d);
b = w(end);
qterm = diag(X*W*X');
q = y.*qterm + ((y*ones(1,d)).*X)*v + y*b;
end
%%
function f = qloss(I,Xtrain,label,w,lam)
f = sum(log(1 + exp(-myquadratic(Xtrain,label,I,w))))/length(I) + 0.5*lam*w'*w;
end
%%
function g = qlossgrad(I,Xtrain,label,w,lam)
aux = exp(-myquadratic(Xtrain,label,I,w));
a = -aux./(1+aux);
X = Xtrain(I,:);
d = size(X,2);
d2 = d^2;
y = label(I);
ya = y.*a;
qterm = X'*((ya*ones(1,d)).*X);
lterm = X'*ya;
sterm = sum(ya);
g = [qterm(:);lterm;sterm]/length(I) + lam*w;
end

function [w,fvals,gnorm] = SGDNest(X,y,w,batch_size,no_epochs,tol)
    fvals = zeros(no_epochs);
    gnorm = zeros(no_epochs);
    alpha = 1e-3;
    lam = 0.001;
    [N,~] = size(X);
    momentum = 0.9;
    last_change = 0;

    for k=1:no_epochs
        Iepoch = randperm(N);
        b = 1;
        while b < N
            curr_ind = Iepoch(b:min(b + batch_size,N));
            proj = w + momentum*last_change;
            g = qlossgrad(curr_ind,X,y,proj,lam);
            last_change = (momentum*last_change) - alpha*g;
            w = w + last_change;
            b = b + batch_size;
        end
        fvals(k) = qloss(1:N,X,y,w,lam);
        gnorm(k) = norm(qlossgrad(1:N,X,y,w,lam));
        if gnorm(k) < tol
            break
        end
        fprintf('k = %d, f = %d, ||g|| = %d\n',k,fvals(k),gnorm(k));
        % reset last change for next epoch
        %last_change=0;
    end
end