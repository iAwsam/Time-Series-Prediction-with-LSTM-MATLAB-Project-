%% Step 1: Generate 1000 Random Sine and Cosine Sequences
numSequences = 1000;
sequenceLength = 100; % Number of time steps per sequence
X = cell(numSequences,1); % Inputs
Y = cell(numSequences,1); % Targets

for i = 1:numSequences
    freq = rand * 5;         % Random frequency
    phase = rand * 2*pi;     % Random phase
    t = linspace(0,1,sequenceLength);
    
    if rand > 0.5
        X{i} = sin(2*pi*freq*t + phase);
    else
        X{i} = cos(2*pi*freq*t + phase);
    end
    Y{i} = X{i}; % We want to predict the same sequence
end

%% Step 2: Prepare Data for LSTM
% Normalize data
mu = mean(cell2mat(X));
sigma = std(cell2mat(X));
X = cellfun(@(x) (x - mu) / sigma, X, 'UniformOutput', false);
Y = X;

%% Step 3: Define LSTM Network
inputSize = 1;
numHiddenUnits = 100;
outputSize = 1;

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(outputSize)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'MiniBatchSize', 64, ...
    'Shuffle','every-epoch', ...
    'Verbose',0, ...
    'Plots','training-progress');

%% Step 4: Train the LSTM Network
net = trainNetwork(X, Y, layers, options);

%% Step 5: Predict on Test Data
% Select a random test sequence (e.g., 1001st sine or cosine)
freq = rand * 5;
phase = rand * 2*pi;
tTest = linspace(0,1,sequenceLength);
testInput = sin(2*pi*freq*tTest + phase); % or cos(...) for cosine

% Normalize test input
testInputNorm = (testInput - mu) / sigma;

% Predict
predictedOutputNorm = predict(net, testInputNorm');

% Denormalize
predictedOutput = predictedOutputNorm * sigma + mu;

% Plot
figure;
plot(tTest, testInput, 'b', 'DisplayName','Actual');
hold on;
plot(tTest, predictedOutput, 'r--', 'DisplayName','Predicted');
legend;
xlabel('Time');
ylabel('Amplitude');
title('Sine/Cosine Wave Prediction with LSTM');
grid on;
