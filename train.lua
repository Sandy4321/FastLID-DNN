require "torch"
require "nn"
require "cunn"

print("Setting up training dataset...")
local feature_dim = 39  -- 13 MFCCs, 13 delta MFCCS, 13 delta-delta MFCCs
local dataset={}
local features_file="/pool001/atitus/FastLID-DNN/data_prep/feats/features_train_labeled"
local dataset_size=0
local labels = {outofset = 1, english = 2, german = 3, mandarin = 4}
local total_utterances=1503
local max_utterances=total_utterances
local current_utterance_count=0
print("Using a total of " .. max_utterances .. " utterances out of " .. total_utterances)

local current_utterance=""
for line in io.lines(features_file) do
    -- Find utterance ID
    local utt = string.sub(line, 1, 4)
    if utt ~= current_utterance then
        -- Check if we should use this utterance
        if current_utterance_count % (total_utterances / max_utterances) == 0 then
            current_utterance = utt
        end
        current_utterance_count = current_utterance_count + 1
    end

    if utt == current_utterance then
        -- Find language label
        local lang_i, lang_j = string.find(line, "[a-z]+ ", 5)
        local lang = string.sub(line, lang_i, lang_j - 1)   -- Cut off trailing whitespace
        local label = labels[lang]

        -- Read in features into a tensor
        local feature_strs = string.sub(line, lang_j + 1)
        local feature_tensor = torch.CudaTensor(feature_dim)
        local feature_idx = 1
        for feature_str in string.gmatch(feature_strs, "[%-]?[0-9]*%.[0-9]*") do
            feature_tensor[feature_idx] = tonumber(feature_str)
            feature_idx = feature_idx + 1
        end

        -- Add this to the dataset
        dataset[dataset_size + 1] = {feature_tensor, label}
        dataset_size = dataset_size + 1
    end
end
function dataset:size() return dataset_size end
print("Done setting up dataset with " .. dataset_size .. " datapoints across " .. max_utterances .. " utterances.")

print("Setting up neural network...")

local inputs = feature_dim
local outputs = 4       -- number of classes (three languages + OOS)
local hidden_units_1 = 256
local hidden_units_2 = 256
local dropout_prob = 0.5

mlp = nn.Sequential();  -- make a multi-layer perceptron

-- First hidden layer with constant bias term and ReLU activation
mlp:add(nn.Linear(inputs, hidden_units_1))
mlp:add(nn.Add(hidden_units_1, true))
mlp:add(nn.ReLU())
mlp:add(nn.Dropout(dropout_prob))

-- Second hidden layer with constant bias term and ReLU activation as well
mlp:add(nn.Linear(hidden_units_1, hidden_units_2))
mlp:add(nn.Add(hidden_units_2, true))
mlp:add(nn.ReLU())
mlp:add(nn.Dropout(dropout_prob))

-- Output layer with softmax layer
mlp:add(nn.Linear(hidden_units_2, outputs))
mlp:add(nn.LogSoftMax())
print("Done setting up neural network.")

-- Convert our network and tensors to CUDA-compatible versions
print("Converting network to CUDA...")
mlp:cuda()
print("Done conversion.")

print("Training neural network...")
-- Use class negative log likelihood (NLL) criterion and stochastic gradient descent to train network
criterion = nn.ClassNLLCriterion()  
criterion.sizeAverage = false   -- Needed since this is in non-batch mode
criterion:cuda()
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.0025
trainer.maxIteration = 100
trainer:train(dataset)
print("Done training neural network.")

print("Saving...")
local net_filename = "/pool001/atitus/FastLID-DNN/models/256_256_dropout"
torch.save(net_filename, mlp)
print("Saved.")
