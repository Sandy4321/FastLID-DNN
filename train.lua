require "torch"
require "nn"
require "optim"

local lre03DatasetReader = require "lre03DatasetReader"

-- Parse command-line options
local opt = lapp[[
   -n,--network       (default "")          reload pretrained network
   -b,--batchSize     (default 128)         batch size
   -e,--epochs        (default 100)         number of epochs in training
   -g,--gpu                                 train on GPU
   --netFilename      (string)              name of file to save network to
   --languages        (string)              languages being used, delimited by "_"
   --earlyStopping                          perform generalization loss-based early stopping
   --bestValidationFER (default 1.0)        best validation frame error rate seen so far
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

local feature_dim = 39
local context_frames = 20
local optimState = {}
if opt.network == '' then
    print("Setting up neural network...")
    -- Use historical frames as context in input vector
    local inputs = feature_dim * (context_frames + 1)

    --outputs = 4       -- number of classes (three languages + OOS)
    outputs = 3       -- number of classes (two languages + OOS)

    local hidden_units_1 = 1024
    local hidden_units_2 = 1024
    local hidden_units_3 = 512
    local hidden_units_4 = 64
    local hidden_units_5 = 1024

    model = nn.Sequential();  -- make a multi-layer perceptron

    -- First hidden layer with constant bias term and ReLU activation
    model:add(nn.Linear(inputs, hidden_units_1))
    model:add(nn.Add(hidden_units_1, true))
    model:add(nn.ReLU())

    -- Second hidden layer with constant bias term and ReLU activation as well
    model:add(nn.Linear(hidden_units_1, hidden_units_2))
    model:add(nn.Add(hidden_units_2, true))
    model:add(nn.ReLU())
    
    -- Third hidden layer with constant bias term and ReLU activation as well
    model:add(nn.Linear(hidden_units_2, hidden_units_3))
    model:add(nn.Add(hidden_units_3, true))
    model:add(nn.ReLU())

    -- Fourth hidden layer with constant bias term and ReLU activation as well
    model:add(nn.Linear(hidden_units_3, hidden_units_4))
    model:add(nn.Add(hidden_units_4, true))
    model:add(nn.ReLU())

    -- Fifth hidden layer with constant bias term and ReLU activation as well
    model:add(nn.Linear(hidden_units_4, hidden_units_5))
    model:add(nn.Add(hidden_units_5, true))
    model:add(nn.ReLU())

    -- Output layer with softmax layer
    model:add(nn.Linear(hidden_units_5, outputs))
    --model:add(nn.Linear(hidden_units_4, outputs))
    model:add(nn.LogSoftMax())
    print("Done setting up neural network.")
else
    print("Loading existing neural network " .. opt.network .. "...")
    model = torch.load(opt.network)
    print("Loaded existing neural network " .. opt.network)
    
    print("Loading existing optimizer state " .. opt.network .. "...")
    local oldOptimStateFilename = opt.network .. "_optimState"
    optimState = torch.load(oldOptimStateFilename)
    print("Loaded existing optimizer state " .. opt.network)
end

model:training()
print("Using model:")
print(model)

print("Using optimizer state:")
print(optimState)

if opt.gpu then
    -- Convert our network to CUDA-compatible version
    print("Convert network to CUDA")
    model = model:cuda()
end

--local lang2label = {outofset = 1, english = 2, german = 3, mandarin = 4}
local lang2label = {outofset = 1, english = 2, german = 1, mandarin = 3}

-- Load the validation dataset
print("Loading validation dataset...")
local validate_file="/pool001/atitus/FastLID-DNN/data_prep/feats/" .. opt.languages .. "_validate"

-- Balance the data
local label2maxframes = torch.zeros(4)
--local min_frames = 22022        -- Count for German, the minimum in this label set
local min_frames = 45490        -- Count for Mandarin, the minimum in this label set
label2maxframes[lang2label["outofset"]] = min_frames
label2maxframes[lang2label["english"]] = min_frames
label2maxframes[lang2label["german"]] = min_frames
label2maxframes[lang2label["mandarin"]] = min_frames
label2maxframes:floor()
local readCfg = {
    features_file = validate_file,
    lang2label = lang2label,
    label2maxframes = label2maxframes,
    include_utts = true,
    gpu = opt.gpu
}
local validate_dataset, validate_label2framecount = lre03DatasetReader.read(readCfg)
print("Loaded validation dataset.")

-- Load the training dataset
print("Loading training dataset...")
local train_file="/pool001/atitus/FastLID-DNN/data_prep/feats/" .. opt.languages .. "_train"

-- Balance the data
local label2maxframes = torch.zeros(4)
local min_frames = 671747   -- Count for German, the minimum in our overall label set
--local min_frames = 1318963  -- Count for Mandarin, the minimum in this label set 
label2maxframes[lang2label["outofset"]] = min_frames
label2maxframes[lang2label["english"]] = min_frames
label2maxframes[lang2label["german"]] = min_frames
label2maxframes[lang2label["mandarin"]] = min_frames
label2maxframes:floor()
local readCfg = {
    features_file = train_file,
    lang2label = lang2label,
    label2maxframes = label2maxframes,
    include_utts = true,
    gpu = opt.gpu
}
local train_dataset, train_label2framecount = lre03DatasetReader.read(readCfg)
print("Loaded training dataset.")

print("Training neural network using minibatch size " .. opt.batchSize .. "...")
-- Use class negative log likelihood (NLL) criterion and stochastic gradient descent to train network
local criterion = nn.ClassNLLCriterion(weights) 
if opt.gpu then
    print("Convert criterion to CUDA")
    criterion = criterion:cuda()
end

-- Set up confusion matrices
--labels = {lang2label["outofset"], lang2label["english"], lang2label["german"], lang2label["mandarin"]}
labels = {lang2label["outofset"], lang2label["english"], lang2label["mandarin"]}
local train_confusion = optim.ConfusionMatrix(labels)
local validate_confusion = optim.ConfusionMatrix(labels)

local learning_rate = 0.001

-- Values suggested by paper
-- https://arxiv.org/pdf/1412.6980.pdf
local adam_config = {
    learningRate = learning_rate,
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 1e-8
}

-- Track validation FER for early stopping
-- Uses Generalization Loss as discussed in
-- http://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf
local best_validation_fer = opt.bestValidationFER
local gl_threshold = 10.0        -- Hand-tuned

-- Mini-batch training with help of
-- https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua
parameters, gradParameters = model:getParameters()
for epoch = 1,opt.epochs do
    local start_time = sys.clock()

    -- Set up for training (i.e. activate Dropout)
    model:training()

    -- Shuffle our data so we don't get mono-language minibatches
    local shuffle = torch.randperm(train_dataset:size())

    -- Run through each of our mini-batches
    for batch_start = 1,train_dataset:size(),opt.batchSize do
        -- Load our samples
        local train_inputs = torch.zeros(opt.batchSize, feature_dim * (context_frames + 1))
        local targets = torch.zeros(opt.batchSize)
        if opt.gpu then
            train_inputs = train_inputs:cuda()
            targets = targets:cuda()
        end

        local train_input_idx = 1
        for sample_idx = batch_start, math.min(batch_start + opt.batchSize - 1, train_dataset:size()) do
            local shuffle_idx = shuffle[sample_idx]
            local data = train_dataset[shuffle_idx]
            local features_tensor = data[1]
            local label = data[2]
            local utt = data[3]

            -- Load current features
            train_inputs[{ train_input_idx, {1, feature_dim} }] = features_tensor

            -- Load context features, if any
            for context = 1, math.min(context_frames, shuffle_idx - 1) do
                local context_data = train_dataset[shuffle_idx - context]
                local context_features_tensor = context_data[1]
                local context_utt = context_data[3] 
                
                -- Don't let another utterance spill over into this one!
                if context_utt == utt then
                    local slice_begin = (context * feature_dim) + 1
                    local slice_end = (context+1)*feature_dim
                    train_inputs[{ train_input_idx, {slice_begin, slice_end} }] = context_features_tensor
                end
            end

            -- Add target label
            targets[train_input_idx] = label
            train_input_idx = train_input_idx + 1
        end

        -- Local function evaluation for gradient descent
        local eval_func = function(x)
            -- Reset gradient
            gradParameters:zero()

            -- Evaluate function for our whole mini-batch
            local output_probs = model:forward(train_inputs)
            local f = criterion:forward(output_probs, targets)

            -- Estimate df/dW
            local df_do = criterion:backward(output_probs, targets)
            model:backward(train_inputs, df_do)

            -- Update train confusion matrix
            for i = 1,opt.batchSize do
                train_confusion:add(output_probs[i], targets[i])
            end

            -- Return f and df/dW
            return f,gradParameters
        end
        
        -- Optimize gradient
        optim.adam(eval_func, parameters, adam_config, optimState)
    end

    -- Print time statistics
    local end_time = sys.clock()
    local elapsed_time = end_time - start_time
    local time_per_sample = elapsed_time / train_dataset:size()
    print("================================")
    print("Epoch " .. epoch .. ":")
    print("  time to learn 1 sample = " .. (time_per_sample * 1000) .. "ms")
    print("  time to learn all " .. train_dataset:size() .. " samples = " .. (elapsed_time * 1000) .. "ms")
    print("================================")

    -- Print confusion matrix and reset
    print("Training confusion matrix:")
    print(train_confusion)
    train_confusion:zero()



    print("Validating neural network...")
    local correct_frames = 0
    local utterance_output_avgs = {}        -- averaged output probabilities
    local utterance_frame_counts = {}       -- current count of probabilities (used for averaging)
    local utterance_labels = {}             -- correct utterance-level label
    local utterance_count = 0

    local start_time = sys.clock()
    
    -- Set up for validation (i.e. deactivate Dropout)
    model:evaluate()

    local validate_input = torch.zeros(feature_dim * (context_frames + 1))
    if opt.gpu then
        -- Convert our input tensor to CUDA-compatible version
        validate_input = validate_input:cuda()
    end

    for i=1,validate_dataset:size() do
        local data = validate_dataset[i]
        local features_tensor = data[1]
        local label = data[2]
        local utt = data[3]

        -- Load current features
        validate_input:zero()
        validate_input[{ {1, feature_dim} }] = features_tensor

        -- Load context features, if any
        for context = 1, math.min(context_frames, i - 1) do
            local context_data = validate_dataset[i - context]
            local context_features_tensor = context_data[1]
            local context_utt = context_data[3]

            -- Don't let another utterance spill over into this one!
            if context_utt == utt then
                local slice_begin = (context * feature_dim) + 1
                local slice_end = (context+1)*feature_dim
                validate_input[{ {slice_begin, slice_end} }] = context_features_tensor
            end
        end

        -- Evaluate this frame
        local output_probs = model:forward(validate_input)
        local confidence, classification_tensor = torch.max(output_probs, 1)
        local classification = classification_tensor[1]
        if classification == label then
            correct_frames = correct_frames + 1
        end
                
        -- Update confusion matrix
        validate_confusion:add(output_probs, label)

        -- Update utterance-level stats
        if utt ~= current_utterance then
            -- Create new entry
            utterance_count = utterance_count + 1
            utterance_output_avgs[utterance_count] = output_probs
            utterance_labels[utterance_count] = label
            utterance_frame_counts[utterance_count] = 1
            current_utterance = utt
        else
            -- Update average
            local old_frame_count = utterance_frame_counts[utterance_count]
            local old_avg = utterance_output_avgs[utterance_count]
            local new_avg = torch.mul(torch.add(output_probs, torch.mul(old_avg, old_frame_count)), 1.0 / (old_frame_count + 1))
            utterance_output_avgs[utterance_count] = new_avg
            utterance_frame_counts[utterance_count] = utterance_frame_counts[utterance_count] + 1
        end
    end

    -- Print time statistics for frame-level validation
    local end_time = sys.clock()
    local elapsed_time = end_time - start_time
    local time_per_sample = elapsed_time / validate_dataset:size()
    local current_validation_fer = 1.0 - (correct_frames / validate_dataset:size())
    print("================================")
    print("Frame-Level Validation:")
    print("  time to validate 1 sample = " .. (time_per_sample * 1000) .. "ms")
    print("  time to validate all " .. validate_dataset:size() .. " samples = " .. (elapsed_time * 1000) .. "ms")
    print("  FER: " .. current_validation_fer)
    print("================================")

    -- Print confusion matrix and reset
    print(validate_confusion)
    validate_confusion:zero()

    local start_time = sys.clock()

    local correct_utterances = 0
    --local max_utterances = 284  -- Total in validation set (English, German, Mandarin)
    --local max_utterances = 211  -- Total in validation set (English, German)
    --local max_utterances = 214  -- Total in validation set (German, Mandarin)
    local max_utterances = 444  -- Total in validation set (English, Mandarin)
    for i=1,max_utterances do
        -- Test whole utterance
        local label = utterance_labels[i]
        local confidence, classification_tensor = torch.max(utterance_output_avgs[i], 1)
        local classification = classification_tensor[1]
        if classification == label then
            correct_utterances = correct_utterances + 1
        end
        
        -- Update confusion matrix
        validate_confusion:add(utterance_output_avgs[i], label)
    end

    -- Print time statistics for utterance-level testing
    local end_time = sys.clock()
    local elapsed_time = end_time - start_time
    local time_per_utterance = elapsed_time / max_utterances
    local current_uer = 1.0 - (correct_utterances / max_utterances)
    print("================================")
    print("Utterance-Level Validation:")
    print("  time to validate 1 utterance = " .. (time_per_utterance * 1000) .. "ms")
    print("  time to validate all " .. max_utterances .. " utterances = " .. (elapsed_time * 1000) .. "ms")
    print("  UER: " .. current_uer)
    print("================================")

    -- Print confusion matrix and reset
    print(validate_confusion)
    validate_confusion:zero()
    print("Done validating neural network.")

    if opt.earlyStopping then
        -- Check if we should stop early
        print("Best validation FER so far " .. best_validation_fer .. ", current FER " .. current_validation_fer)
        if current_validation_fer < best_validation_fer then
            best_validation_fer = current_validation_fer
        end
        local generalization_loss = 100.0 * ((current_validation_fer / best_validation_fer) - 1)

        print("Generalization loss:")
        print(generalization_loss)

        if generalization_loss > gl_threshold then
            print("============================")
            print("STOPPING EARLY")
            print("============================")
            break
        end 
    end

    -- Save the network for future training or testing
    print("Saving current network state...")
    torch.save(opt.netFilename, model)
    print("Saved.")
    
    -- Save the optimizer state so we can pick up from a checkpoint later!
    print("Saving current optimizer state...")
    local optimStateFilename = opt.netFilename .. "_optimState"
    torch.save(optimStateFilename, optimState)
    print("Saved.")
end

print("Done training neural network.")
