clear
clc

train_data = readtable('Train_SU63ISt.csv','Delimiter',',');
test_data = readtable('Test_0qrQsBZ.csv','Delimiter',',');
%%
summary(train_data)

train_data(1:5,:)
%%
train_data.Datetime=datenum([train_data.Datetime],'dd-mm-yyyy HH:MM');
test_data.Datetime=datenum([test_data.Datetime],'dd-mm-yyyy HH:MM');
plot(train_data.Datetime,train_data.Count)
datetick('x','keepticks','keeplimits')
%%
ts=train_data.Datetime;
ts=ts*24*60*60;
ts=ts-ts(1);
train_data.Datetime = ts;
plot(ts,train_data.Count)

%% Detrending 
% vectors t,x are given (time and the signal)
degree = 5; % to be adjusted
[p, S, mu] = polyfit(train_data.Datetime,train_data.Count, degree);
xp = polyval(p,train_data.Datetime, [], mu);
q = train_data.Count - xp;
plot(train_data.Datetime,train_data.Count,'r')
hold on
plot(train_data.Datetime,xp,'k')
hold on
plot(train_data.Datetime,q,'b')

%% Removing Seasonality
% Tot=length(train_data.Datetime);
% s = 24;
% sidx = cell(s,1); % Preallocation
% 
% for i = 1:Tot/s
%  sidx{i,1} = (i-1)*s+1:i*s;
% end
% 
% % sidx{1:2}
% 
% sst = cellfun(@(x) mean(q(x)),sidx);
% 
% %% Put smoothed values back into a vector of length N
% sstm = repmat(sst,1,24)';
% sstm=sstm(:);
% % Center the seasonal estimate (additive)
% sBar = mean(sstm); % for centering
% sstm = sstm-sBar;
% 
% figure
% plot(sstm)
% title 'Stable Seasonal Component';
% %%
% dt = q - sstm;
% 
% figure
% plot(dt)
% title 'Deseasonalized Series';

%%
%   y - feedback time series.

T = tonndata(y,false,false);

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% Create a Nonlinear Autoregressive Network
feedbackDelays = 1:5;
hiddenLayerSize = 10;
net = narnet(feedbackDelays,hiddenLayerSize,'open',trainFcn);

% Choose Feedback Pre/Post-Processing Functions
% Settings for feedback input are automatically applied to feedback output
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};

% Prepare the Data for Training and Simulation
% The function PREPARETS prepares timeseries data for a particular network,
% shifting time by the minimum amount to fill input states and layer
% states. Using PREPARETS allows you to keep your original time series data
% unchanged, while easily customizing it for networks with differing
% numbers of delays, with open loop or closed loop feedback modes.
[x,xi,ai,t] = preparets(net,{},{},T);

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivision
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'time';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate', 'ploterrhist', ...
    'plotregression', 'plotresponse', 'ploterrcorr', 'plotinerrcorr'};

% Train the Network
[net,tr] = train(net,x,t,xi,ai);

% Test the Network
y = net(x,xi,ai);
e = gsubtract(t,y);
performance = perform(net,t,y)

% Recalculate Training, Validation and Test Performance
trainTargets = gmultiply(t,tr.trainMask);
valTargets = gmultiply(t,tr.valMask);
testTargets = gmultiply(t,tr.testMask);
trainPerformance = perform(net,trainTargets,y)
valPerformance = perform(net,valTargets,y)
testPerformance = perform(net,testTargets,y)

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotregression(t,y)
%figure, plotresponse(t,y)
%figure, ploterrcorr(e)
%figure, plotinerrcorr(x,e)

% Closed Loop Network
% Use this network to do multi-step prediction.
% The function CLOSELOOP replaces the feedback input with a direct
% connection from the outout layer.
netc = closeloop(net);
netc.name = [net.name ' - Closed Loop'];
view(netc)
[xc,xic,aic,tc] = preparets(netc,{},{},T);
yc = netc(xc,xic,aic);
closedLoopPerformance = perform(net,tc,yc)

% Multi-step Prediction
% Sometimes it is useful to simulate a network in open-loop form for as
% long as there is known data T, and then switch to closed-loop to perform
% multistep prediction. Here The open-loop network is simulated on the
% known output series, then the network and its final delay states are
% converted to closed-loop form to produce predictions for 5 more
% timesteps.
[x1,xio,aio,t] = preparets(net,{},{},T);
[y1,xfo,afo] = net(x1,xio,aio);
[netc,xic,aic] = closeloop(net,xfo,afo);
[y2,xfc,afc] = netc(cell(0,5112),xic,aic);
%%
results = cell2mat(y2);
test_trend = polyval(p,test_data.Datetime, [], mu);
test_data.Count = results'+test_trend;
test_data.Datetime = [];
%%
test_data.Properties.VariableNames={'ID', 'Count'};
writetable(test_data,'submission3.csv','Delimiter',',');