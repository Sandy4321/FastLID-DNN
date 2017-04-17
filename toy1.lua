require "torch"
require "nn"

-- TODO: get this set up with actual, real data
print("Setting up dataset...")
local feature_dim = 38  -- 12 MFCCs, 12 delta MFCCS, 12 delta-delta MFCCs, delta power, delta-delta power
dataset={}
function dataset:size() return 1 end -- 1 example (for now)
features = torch.randn(feature_dim)
class = 1       -- Completely arbitrary
dataset[1] = {features, class}
print("Done setting up dataset.")

print("Setting up neural network...")

local inputs = feature_dim
local outputs = 4       -- arbitrary for now
local hidden_units_1 = 1024     -- also arbitrary
local hidden_units_2 = 1024     -- also arbitrary

mlp = nn.Sequential();  -- make a multi-layer perceptron

-- First hidden layer with constant bias term and ReLU activation
mlp:add(nn.Linear(inputs, hidden_units_1))
mlp:add(nn.Add(hidden_units_1, true))
mlp:add(nn.ReLU())

-- Second hidden layer with constant bias term and ReLU activation as well
mlp:add(nn.Linear(hidden_units_1, hidden_units_2))
mlp:add(nn.Add(hidden_units_2, true))
mlp:add(nn.ReLU())

-- Output layer with softmax layer
mlp:add(nn.Linear(hidden_units_2, outputs))
mlp:add(nn.SoftMax())

print("Done setting up neural network.")

print("Training neural network...")
-- Use cross entropy criterion and stochastic gradient descent to train network
criterion = nn.CrossEntropyCriterion()  
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 100
trainer:train(dataset)
print("Done training neural network.")

print("Testing neural network...")
-- TODO: actually run on real test data
local confidences = mlp:forward(features)
local confidence, classification_tensor = torch.max(confidences, 1)
local classification = classification_tensor[1]
print("  Testing on training data:")
print("    Expected:")
print(class)
print("    Got:")
print(classification)
print("  Confidences:")
print(confidences)
print("Done testing neural network.")
