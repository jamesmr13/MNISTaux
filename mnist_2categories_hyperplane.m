function mnist_2categories_hyperplane()
close all
fsz = 20;
%% Pick the number of PCAs for the representation of images
nPCA = 3;
%%
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
    axis off
end
%% plot some data from category 2
figure; colormap gray
for j = 1:20
    subplot(4,5,j);
    imagesc(train2(:,:,j));
    axis off
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
for j = 1 : n1test
    aux = test1(:,:,j);
    Xtest1(j,:) = aux(:)';
end
for j = 1 :n2test
    aux = test2(:,:,j);
    Xtest2(j,:) = aux(:)';
end
Xtest = [Xtest1;Xtest2]*U(:,1:nPCA);
%% category 1 (1): label 1; category 2 (7): label -1
label = ones(Ntrain,1);
label(n1train+1:Ntrain) = -1;
%% dividing hyperplane: w'*x + b
dim = nPCA;
Y = (label*ones(1,dim + 1)).*[Xtrain,ones(size(Xtrain,1),1)]; % Y*w is the test fucntion
%% optimize w and b using a smooth loss function and SINewton
lam = 0.001; % Tikhonov regularization parameter
fun = @(I,w)fun0(I,Y,w,lam);
gfun = @(I,w)gfun0(I,Y,w,lam);
Hvec = @(I,w,v)Hvec0(I,Y,w,v,lam);
w = ones(dim+1,1);
% params for SINewton
frac = 10;
bsz = ceil(Ntrain/frac); % batch size
kmax = 1e3*frac; % the max number of iterations
tol = 1e-4;
% call the optimizer
[w,f,gnorm] = SINewton(fun,gfun,Hvec,Y,w,bsz,kmax,tol);
wvec = w(1:dim);
b = w(end);
% plot the objective function
figure;
plot(f,'Linewidth',2);
xlabel('iter','fontsize',fsz);
ylabel('f','fontsize',fsz);
set(gca,'fontsize',fsz,'Yscale','log');
% plot the norm of the gradient
figure;
plot(gnorm,'Linewidth',2);
xlabel('iter','fontsize',fsz);
ylabel('||g||','fontsize',fsz);
set(gca,'fontsize',fsz,'Yscale','log');

%% apply the results to the test set
Ntest = n1test+n2test;
testlabel = ones(Ntest,1);
testlabel(n1test+1:Ntest) = -1;
test = testlabel.*(Xtest*wvec + b);
hits = find(test > 0);
misses = find(test < 0);
nhits = length(hits);
nmisses = length(misses);
fprintf('n_correct = %d, n_wrong = %d, accuracy %d percent\n',nhits,nmisses,nhits/Ntest);
%% plot the dividing surface if nPCA = 3
if dim == 3
    xmin = min(Xtrain(:,1)); xmax = max(Xtrain(:,1));
    ymin = min(Xtrain(:,2)); ymax = max(Xtrain(:,2));
    zmin = min(Xtrain(:,3)); zmax = max(Xtrain(:,3));
    nn = 50;
    figure(figPCA);
    [xx,yy,zz] = meshgrid(linspace(xmin,xmax,nn),linspace(ymin,ymax,nn),...
        linspace(zmin,zmax,nn));
    plane = w(1)*xx+w(2)*yy+w(3)*zz+w(4);
    p = patch(isosurface(xx,yy,zz,plane,0));
    p.FaceColor = 'cyan';
    p.EdgeColor = 'none';
    camlight 
    lighting gouraud
    alpha(0.3);
end
end
%%
%% The objective function
function f = fun0(I,Y,w,lam) 
f = sum(log(1 + exp(-Y(I,:)*w)))/length(I) + 0.5*lam*w'*w;
end
%% The gradient of the objective function
function g = gfun0(I,Y,w,lam) 
aux = exp(-Y(I,:)*w);
d1 = size(Y,2);
g = sum(-Y(I,:).*((aux./(1 + aux))*ones(1,d1)),1)'/length(I) + lam*w;
end
%% The Hessian of the objective function times vector v
function Hv = Hvec0(I,Y,w,v,lam) % the Hessian of the objective function times vector v
aux = exp(-Y(I,:)*w);
d1 = size(Y,2);
Hv = sum(Y(I,:).*((aux.*(Y(I,:)*v)./((1+aux).^2)).*ones(1,d1)),1)' + lam*v;
end


