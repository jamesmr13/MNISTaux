function SaveMNIST2mat()
imgFile = 'train-images.idx3-ubyte';
labelFile = 'train-labels.idx1-ubyte';
readDigits = 60000;
offset = 0;
[imgs_train, labels_train] = readMNIST(imgFile, labelFile, readDigits, offset);
labels_train = categorical(labels_train);
%
imgFile = 't10k-images.idx3-ubyte';
labelFile = 't10k-labels.idx1-ubyte';
readDigits = 10000;
offset = 0;
[imgs_test, labels_test] = readMNIST(imgFile, labelFile, readDigits, offset);
labels_test = categorical(labels_test);
save('mnist.mat','imgs_test','imgs_train','labels_test','labels_train');
end