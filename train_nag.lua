require "torch"
require "nn"
require "cunn"
require "optim"

print("Setting up training dataset...")
local feature_dim = 39  -- 13 MFCCs, 13 delta MFCCS, 13 delta-delta MFCCs
local dataset={}
local features_file="/pool001/atitus/FastLID-DNN/data_prep/feats/features_train_labeled"
local dataset_size = 0
local lang2label = {outofset = 1, english = 2, german = 3, mandarin = 4}
local total_utterances = 1503
local max_utterances=total_utterances
print("Using a total of " .. max_utterances .. " utterances out of " .. total_utterances)

local current_utterance=""
local utterances_used = 0
local utterances_seen = 0
for line in io.lines(features_file) do
    -- Find utterance ID
    local utt = string.sub(line, 1, 4)
    if utt ~= current_utterance then
        -- Check if we should use this utterance
        if utterances_seen % (total_utterances / max_utterances) == 0 then
            utterances_used = utterances_used + 1
            current_utterance = utt
        end
        utterances_seen = utterances_seen + 1
    end

    -- Check if we should bail
    if utterances_used > max_utterances then
        break
    end

    if utt == current_utterance then
        -- Find language label
        local lang_i, lang_j = string.find(line, "[a-z]+ ", 5)
        local lang = string.sub(line, lang_i, lang_j - 1)   -- Cut off trailing whitespace
        local label = lang2label[lang]

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
local hidden_units_1 = 1024
local hidden_units_2 = 1024
local dropout_prob = 0.5

model = nn.Sequential();  -- make a multi-layer perceptron

-- First hidden layer with constant bias term and ReLU activation
model:add(nn.Linear(inputs, hidden_units_1))
model:add(nn.Add(hidden_units_1, true))
model:add(nn.ReLU())
model:add(nn.Dropout(dropout_prob))

-- Second hidden layer with constant bias term and ReLU activation as well
model:add(nn.Linear(hidden_units_1, hidden_units_2))
model:add(nn.Add(hidden_units_2, true))
model:add(nn.ReLU())
model:add(nn.Dropout(dropout_prob))

-- Output layer with softmax layer
model:add(nn.Linear(hidden_units_2, outputs))
model:add(nn.LogSoftMax())
print("Done setting up neural network.")

-- Convert our network and tensors to CUDA-compatible versions
print("Converting network to CUDA...")
model:cuda()
print("Done conversion.")

local batch_size = 512
print("Set batch size to " .. batch_size)

print("Training neural network...")
-- Use class negative log likelihood (NLL) criterion and stochastic gradient descent to train network
local criterion = nn.ClassNLLCriterion()  
criterion:cuda()

-- Set up confusion matrix
local labels = {1, 2, 3, 4}
local confusion = optim.ConfusionMatrix(labels)

-- Train via Nesterov-accelerated gradient descent
-- Mini-batch training with help of
-- https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua
parameters, gradParameters = model:getParameters()
local epochs = 100
for epoch = 1,epochs do
    local start_time = sys.clock()

    -- Run through each of our mini-batches
    for batch_start = 1,dataset:size(),batch_size do
        -- Load our samples
        local inputs = torch.CudaTensor(batch_size, feature_dim)
        local targets = torch.CudaTensor(batch_size)
        local input_idx = 1
        for sample_idx = batch_start, math.min(batch_start + batch_size - 1, dataset:size()) do
            local data = dataset[sample_idx]
            local features_tensor = data[1]
            local label = data[2]

            inputs[input_idx] = features_tensor
            targets[input_idx] = label

            input_idx = input_idx + 1
        end

        -- Local function evaluation for gradient descent
        local eval_func = function(x)
            -- Reset gradient
            gradParameters:zero()

            -- Evaluate function for our whole mini-batch
            local output_probs = model:forward(inputs)
            local f = criterion:forward(output_probs, targets)

            -- Estimate df/dW
            local df_do = criterion:backward(output_probs, targets)
            model:backward(inputs, df_do)

            -- Update confusion matrix
            for i = 1,batch_size do
                confusion:add(output_probs[i], targets[i])
            end

            -- Return f and df/dW
            return f,gradParameters
        end

        -- Optimize gradient
        local nag_config = {
            learningRate = 0.001,
            learningRateDecay = 5e-7,
            momentum = 0.5
        }
        optim.nag(eval_func, parameters, nag_config)
    end

    -- Print time statistics
    local end_time = sys.clock()
    local elapsed_time = end_time - start_time
    local time_per_sample = elapsed_time / dataset:size()
    print("================================")
    print("Epoch " .. epoch .. ":")
    print("  time to learn 1 sample = " .. (time_per_sample * 1000) .. "ms")
    print("  time to learn all " .. dataset:size() .. " samples = " .. (elapsed_time * 1000) .. "ms")
    print("================================")

    -- Print confusion matrix and reset
    print(confusion)
    confusion:zero()

    print("Saving current network state...")
    local net_filename = "/pool001/atitus/FastLID-DNN/models/1k_1k"
    torch.save(net_filename, model)
    print("Saved.")
end
print("Done training neural network.")
