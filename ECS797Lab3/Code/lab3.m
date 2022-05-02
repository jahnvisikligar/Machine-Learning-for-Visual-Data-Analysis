%% information
% facial age estimation
% regression method: linear regression

%% settings
clear;
clc;

% path 
database_path = './data_age.mat';
result_path = './results/';

% initial states
absTestErr = 0;
cs_number = 0;


% cumulative error level
err_level = 5;

%% Training 
load(database_path);

nTrain = length(trData.label); % number of training samples
nTest  = length(teData.label); % number of testing samples
xtrain = trData.feat; % feature
ytrain = trData.label; % labels

w_lr = regress(ytrain,xtrain);
   
%% Testing
xtest = teData.feat; % feature
ytest = teData.label; % labels

yhat_test = xtest * w_lr;

%% Compute the MAE and CS value (with cumulative error level of 5) for linear regression 

MAE = sum(abs(yhat_test - ytest))/(size(ytest,1))

a = abs(yhat_test - ytest);
CS = size(find(a<= err_level))/size(a) * 100

%% generate a cumulative score (CS) vs. error level plot by varying the error level from 1 to 15. The plot should look at the one in the Week6 lecture slides
values = [];
err_lvl = [];
for i = 1:15
    cum_score = size(find(a<= i))/size(a);
    values = [values, cum_score];
    err_lvl = [err_lvl, i];
end

% plot CS vs Error Level
figure(1)
plot(err_lvl, values)
xlabel('Error Level')
ylabel('Cumulative Score')
title('Cumulative Score (CS) vs. Error Level plot')
grid on
grid minor

%% Compute the MAE and CS value (with cumulative error level of 5) for both partial least square regression and the regression tree model by using the Matlab built in functions.

%partial least square regression
mae_plsr = [];
cs_plsr = [];

components = [];
for i = 1:15
    [XL,YL,XS,YS,BETA] = plsregress(xtrain, ytrain, i);
    yhat_test_plsr = [ones(size(xtest,1),1),xtest]*BETA;
    mae_plsr = [mae_plsr, sum(abs(yhat_test_plsr - ytest))/(size(ytest,1))];
    
    b = abs(yhat_test_plsr - ytest);
    cs = size(find(b<= err_level))/size(b) * 100;
    cs_plsr = [cs_plsr, cs];
    
    components = [components, i];
end

%PLOT MAE plsr
figure(2)
plot(components, mae_plsr)
xlabel('Num of components')
ylabel('MAE PLSR')
grid on
grid minor

%PLOT CS plsr
figure(3)
plot(components, cs_plsr)
xlabel('Num of components')
ylabel('CS PLSR')
grid on
grid minor

% best MAE from ncomps
mae_lsr = min(mae_plsr)
% corresponding CS
cs_lsr = cs_plsr(find(mae_plsr == mae_lsr))

% Regression tree model

mae_rt = [];
cs_rt = [];

reg_tree = fitrtree(xtrain, ytrain);
yhat_test_reg = predict(reg_tree, xtest);
mae_rt = [sum(abs(yhat_test_reg - ytest))/(size(ytest,1))]
    
c = abs(yhat_test_reg - ytest);
cs_rt = size(find(c<= err_level))/size(c) * 100

%% Compute the MAE and CS value (with cumulative error level of 5) for Support Vector Regression by using LIBSVM toolbox

mae_svm = [];
cs_svm = [];

svm = fitrsvm(xtrain, ytrain);
yhat_test_svm = predict(svm, xtest);
mae_svm = [sum(abs(yhat_test_svm - ytest))/(size(ytest,1))]
    
d = abs(yhat_test_svm - ytest);
cs_svm = size(find(d<= err_level))/size(d) * 100

% table displaying MAE and CS values and Algorithms

values = {'Linear Regression' MAE CS; 'PLS Regression' mae_lsr cs_lsr; 'Regression Tree' mae_rt cs_rt; 'Support Vector Regression' mae_svm cs_svm};
T = cell2table(values, 'VariableNames',{'Algorithm' 'MAE' 'CS'})
