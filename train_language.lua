require "torch"
require "nn"
require "cunn"
require "optim"

-- Parse command-line options
local opt = lapp[[
   -n,--network       (default "")          reload pretrained network
   -f,--full                                use the full training dataset
   -o,--optimization  (default "")          optimization: Adam | NAG 
   -b,--batchSize     (default 128)         batch size
   -e,--epochs        (default 100)         number of epochs in training
   -g,--gpu                                 train on GPU
   --netFilename      (string)              name of file to save network to
   -t,--threads       (default 4)           number of threads
   -l,--language      (string)              language to train network for
]]

-- Fix seed
torch.manualSeed(1)

-- Threads
torch.setnumthreads(opt.threads)
print('Set nb of threads to ' .. torch.getnumthreads())

print("Setting up training dataset...")
local dataset_full={}
local features_file="/pool001/atitus/FastLID-DNN/data_prep/feats/features_train_labeled"
local dataset_full_size = 0
local lang2label = {outofset = 1, english = 1, german = 1, mandarin = 1}
lang2label[opt.language] = 2    -- Only language we want to classify
local label2uttcount = torch.zeros(2)
local total_utterances = 1503
local in_set_frames = 0

-- Only use full dataset if we say so
local max_utterances = 100
if opt.full then
    max_utterances = total_utterances
end
print("Using a total of " .. max_utterances .. " utterances out of " .. total_utterances)

local feature_dim = 39  -- 13 MFCCs, 13 delta MFCCS, 13 delta-delta MFCCs

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
        label2uttcount[label] = label2uttcount[label] + 1
        if label == 2 then
            -- In-class; add to our count
            in_set_frames = in_set_frames + 1
        end

        -- Read in features into a tensor
        local feature_strs = string.sub(line, lang_j + 1)
        local feature_tensor = torch.zeros(feature_dim)
        if opt.gpu then
            feature_tensor = feature_tensor:cuda()
        end
        local feature_idx = 1
        for feature_str in string.gmatch(feature_strs, "[%-]?[0-9]*%.[0-9]*") do
            feature_tensor[feature_idx] = tonumber(feature_str)
            feature_idx = feature_idx + 1
        end

        -- Add this to the dataset
        dataset_full[dataset_full_size + 1] = {feature_tensor, label}
        dataset_full_size = dataset_full_size + 1
    end
end
print("Finished loading dataset with " .. dataset_full_size .. " frames: " .. in_set_frames .. " in set, " .. (dataset_full_size - in_set_frames) .. " out of set")

-- Downsample OOS data to prevent unbalanced data
local dataset = {}
local dataset_size = 0
local oos_frames = 0
for i = 1,dataset_full_size do
    local full_data = dataset_full[i]
    local label = full_data[2]

    if label == 1 then
        -- OOS: add to count
        oos_frames = oos_frames + 1

        if oos_frames <= in_set_frames then
            dataset_size = dataset_size + 1
            dataset[dataset_size] = dataset_full[i]
        end
    else
        -- In set: always add
        dataset_size = dataset_size + 1
        dataset[dataset_size] = dataset_full[i]
    end
end
function dataset:size() return dataset_size end
print("Done setting up balanced dataset with " .. dataset_size .. " datapoints")

local context_frames = 10
if opt.network == '' then
    print("Setting up neural network...")
    -- Use historical frames as context in input vector
    local inputs = feature_dim * (context_frames + 1)
    local outputs = 2       -- number of classes (one language + OOS)
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
else
    print("Loading existing neural network " .. opt.network .. "...")
    model = torch.load(opt.network)
    print("Loaded existing neural network " .. opt.network)
end

-- Set up for training (i.e. activate Dropout)
model:training()
print("Using model:")
print(model)

if opt.gpu then
    -- Convert our network to CUDA-compatible version
    print("Converting network to CUDA")
    model = model:cuda()
end

print("Training neural network using minibatch size " .. opt.batchSize .. "...")
-- Use class negative log likelihood (NLL) criterion and stochastic gradient descent to train network

-- Weights data to account for class imbalance in NIST 2003 dataset
-- local weights = torch.cdiv(torch.ones(label2uttcount:size(1)), label2uttcount)
-- if opt.gpu then
--     print("Converting weights to CUDA")
--     weights = weights:cuda()
-- end
-- local criterion = nn.ClassNLLCriterion(weights) 
local criterion = nn.ClassNLLCriterion() 
if opt.gpu then
    print("Converting criterion to CUDA")
    criterion = criterion:cuda()
end
-- print("Using class NLL criterion with weights:")
-- print(weights)

-- Set up confusion matrix
local labels = {1, 2}
local confusion = optim.ConfusionMatrix(labels)

-- Values suggested by paper
-- https://arxiv.org/pdf/1412.6980.pdf
local adam_config = {
    learningRate = 0.001,
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 1e-8
}

-- Nesterov-accelerated gradient descent
local nag_config = {
    learningRate = 0.001,
    momentum = 0.5
}

-- Mini-batch training with help of
-- https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua
parameters, gradParameters = model:getParameters()
for epoch = 1,opt.epochs do
    local start_time = sys.clock()

    -- Shuffle our data so we don't get mono-language minibatches
    local shuffle = torch.randperm(dataset:size())

    -- Run through each of our mini-batches
    for batch_start = 1,dataset:size(),opt.batchSize do
        -- Load our samples
        local inputs = torch.zeros(opt.batchSize, feature_dim * (context_frames + 1))
        local targets = torch.zeros(opt.batchSize)
        if opt.gpu then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end

        local input_idx = 1
        for sample_idx = batch_start, math.min(batch_start + opt.batchSize - 1, dataset:size()) do
            local data = dataset[shuffle[sample_idx]]
            local features_tensor = data[1]
            local label = data[2] 

            -- Load current features
            inputs[{ input_idx, {1, feature_dim} }] = features_tensor

            -- Load context features, if any
            for context = 1, math.min(context_frames, input_idx - 1) do
                local context_data = dataset[shuffle[sample_idx - context]]
                local context_features_tensor = context_data[1]
                local context_label = context_data[2] 
                local slice_begin = (context * feature_dim) + 1
                local slice_end = (context+1)*feature_dim
                inputs[{ input_idx - context, {slice_begin, slice_end} }] = context_features_tensor
            end

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
            for i = 1,opt.batchSize do
                confusion:add(output_probs[i], targets[i])
            end

            -- Return f and df/dW
            return f,gradParameters
        end
        
        -- Optimize gradient
        if opt.optimization == "Adam" then
            optim.adam(eval_func, parameters, adam_config)
        elseif opt.optimization == "NAG" then
            optim.nag(eval_func, parameters, nag_config)
        else
            error("Unknown optimization method " .. opt.optimization)
        end
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
    torch.save(opt.netFilename, model)
    print("Saved.")
end
print("Done training neural network.")
