function test

dbstop if error;
addpath('.\tool');
train_images = loadMNISTImages('train-images.idx3-ubyte');
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');
test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
train_images = train_images(:,1:10000);
train_labels = train_labels(1:10000);
testData = test_images(:,1:1000);
testLabel = test_labels(1:1000);
% numLabels = unique(train_labels);
% numTest = size(testData,2);
%# train one-against-all models
%model = cell(numel(numLabels),1);
model = svmtrain(train_labels,train_images', '-c 1  -g 0.2 -b 1');
[~,~,p] = svmpredict(testLabel, testData', model, '-b 1');
[~,pred] = max(p,[],2);
print "accuracy:";
acc = sum(pred == testLabel) ./ numel(testLabel)   %# accuracy
C = confusionmat(testLabel, pred);                   %# confusion matrix

% for k=1:numel(numLabels)
%     model{k} = svmtrain(double(train_labels==(k-1)),train_images', '-c 1  -g 0.2 -b 1');
% end
% 
% %# get probability estimates of test instances using each model
% prob = zeros(numTest,numel(numLabels));
% for k=1:numel(numLabels)
%     [~,~,p] = svmpredict(double(testLabel==(k-1)), testData', model{k}, '-b 1');
%     prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k
% end
% 
% %# predict the class with the highest probability




end