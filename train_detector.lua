require "torch"
require "nn"
require "optim"

local lre03DatasetReader = require "lre03DatasetReader"

-- Detect whether language spoken is in or out of set

-- Parse command-line options
local opt = lapp[[
   -n,--network       (default "")          reload pretrained network
   -d,--dropout                             use dropout (50%) while training
   -o,--optimization  (default "")          optimization: Adam | NAG 
   -b,--batchSize     (default 128)         batch size
   -e,--epochs        (default 100)         number of epochs in training
   -g,--gpu                                 train on GPU
   --netFilename      (string)              name of file to save network to
   -t,--threads       (default 4)           number of threads
]]

if opt.gpu then
    require "cunn"
end

-- Fix seed
torch.manualSeed(1)

-- Threads
torch.setnumthreads(opt.threads)
print('Set nb of threads to ' .. torch.getnumthreads())

local features_file="/pool001/atitus/FastLID-DNN/data_prep/feats/features_train_labeled"
local lang2label = {outofset = 1, english = 2, german = 3, mandarin = 4}

-- Only use full dataset if we say so
local total_frames = 469083 
local label2maxframes = torch.zeros(4)
    
-- Balance the data
local min_frames = 111277   -- Count for German, the minimum in this label set
label2maxframes[lang2label["outofset"]] = min_frames
label2maxframes[lang2label["english"]] = min_frames
label2maxframes[lang2label["german"]] = min_frames
label2maxframes[lang2label["mandarin"]] = min_frames
label2maxframes:floor()

-- Load the training dataset
local feature_dim = 39
local context_frames = 10
local readCfg = {
    features_file = features_file,
    lang2label = lang2label,
    label2maxframes = label2maxframes,
    include_utts = true,
    gpu = opt.gpu
}
local dataset, label2uttcount = lre03DatasetReader.read(readCfg)

if opt.network == '' then
    print("Setting up neural network...")
    -- Use historical frames as context in input vector
    local inputs = feature_dim * (context_frames + 1)
    outputs = 2         -- in-set vs. out-of-set
    local hidden_units_1 = 1024
    local hidden_units_2 = 512
    local hidden_units_3 = 128
    local dropout_prob = 0.5

    model = nn.Sequential();  -- make a multi-layer perceptron

    -- First hidden layer with constant bias term and ReLU activation
    model:add(nn.Linear(inputs, hidden_units_1))
    model:add(nn.Add(hidden_units_1, true))
    model:add(nn.ReLU())
    if opt.dropout then
        model:add(nn.Dropout(dropout_prob))
    end

    -- Second hidden layer with constant bias term and ReLU activation as well
    model:add(nn.Linear(hidden_units_1, hidden_units_2))
    model:add(nn.Add(hidden_units_2, true))
    model:add(nn.ReLU())
    if opt.dropout then
        model:add(nn.Dropout(dropout_prob))
    end
    
    -- Third hidden layer with constant bias term and ReLU activation as well
    model:add(nn.Linear(hidden_units_2, hidden_units_3))
    model:add(nn.Add(hidden_units_3, true))
    model:add(nn.ReLU())
    if opt.dropout then
        model:add(nn.Dropout(dropout_prob))
    end

    -- Output layer with softmax layer
    model:add(nn.Linear(hidden_units_3, outputs))
    --model:add(nn.Linear(hidden_units_2, outputs))
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
    print("Convert network to CUDA")
    model = model:cuda()
end

print("Training neural network using minibatch size " .. opt.batchSize .. "...")
-- Use class negative log likelihood (NLL) criterion and stochastic gradient descent to train network
-- Weights data to account for class imbalance in NIST 2003 dataset
-- local weights = torch.cdiv(torch.ones(label2uttcount:size(1)), label2uttcount)
-- if opt.gpu then
--     print("Convert weights to CUDA")
--     weights = weights:cuda()
-- end
-- local criterion = nn.ClassNLLCriterion(weights) 
local criterion = nn.ClassNLLCriterion() 
if opt.gpu then
    print("Convert criterion to CUDA")
    criterion = criterion:cuda()
end
-- print("Using class NLL criterion with weights:")
-- print(weights)

-- Set up confusion matrix
labels = {1, 2}
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
            local shuffle_idx = shuffle[sample_idx]
            local data = dataset[shuffle_idx]
            local features_tensor = data[1]
            local label = data[2]
            if label >= 2 then
                label = 2       -- All languages in-set marked as such
            end
            local utt = data[3]

            -- Load current features
            inputs[{ input_idx, {1, feature_dim} }] = features_tensor

            -- Load context features, if any
            for context = 1, math.min(context_frames, shuffle_idx - 1) do
                local context_data = dataset[shuffle_idx - context]
                local context_features_tensor = context_data[1]
                local context_utt = context_data[3] 
                
                -- Don't let another utterance spill over into this one!
                if context_utt == utt then
                    local slice_begin = (context * feature_dim) + 1
                    local slice_end = (context+1)*feature_dim
                    inputs[{ input_idx, {slice_begin, slice_end} }] = context_features_tensor
                end
            end

            -- Add target label
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
