require "torch"
require "nn"

print("Setting up training dataset...")
local feature_dim = 39  -- 13 MFCCs, 13 delta MFCCS, 13 delta-delta MFCCs
local dataset={}
local features_file="/pool001/atitus/FastLID-DNN/data_prep/feats/features_train_labeled"
local dataset_size=0
local truncated_dataset_size = 20
local labels = {outofset = 1, english = 2, german = 3, mandarin = 4}
for line in io.lines(features_file) do
    -- Find utterance ID
    local utt = string.sub(line, 1, 4)

    -- Find language label
    local lang_i, lang_j = string.find(line, "[a-z]+ ", 5)
    local lang = string.sub(line, lang_i, lang_j - 1)   -- Cut off trailing whitespace

    -- Read in features into a tensor
    local feature_strs = string.sub(line, lang_j + 1)
    local feature_tensor = torch.zeros(feature_dim)
    local feature_idx = 1
    for feature_str in string.gmatch(feature_strs, "[%-]?[0-9]*%.[0-9]*") do
        feature_tensor[feature_idx] = tonumber(feature_str)
        feature_idx = feature_idx + 1
    end

    -- Add this to the dataset
    dataset[dataset_size + 1] = {feature_tensor, labels[lang]}
    dataset_size = dataset_size + 1

    -- Only use part of the dataset for this toy example!
    if dataset_size >= truncated_dataset_size then
        break
    end
end
function dataset:size() return dataset_size end
print("Done setting up dataset.")

print("Setting up neural network...")

local inputs = feature_dim
local outputs = 4       -- number of classes (three languages + OOS)
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
