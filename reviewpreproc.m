%after importing data. reviews as cell array x, classes as vector y
xfeat=featurize_bigram(x,2,1,1);
%lowercase, porter stem, remove stop words

[xtrain,xtest,ytrain,ytest]=makeholdoutset(xfeat,y,20);

csvwrite('revxtrain.csv',xtrain);
csvwrite('revytrain.csv',ytrain);
csvwrite('revytest.csv',ytest+1);
csvwrite('revxtest.csv',xtest);