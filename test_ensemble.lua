require "torch"
require "nn"
require "cunn"
require "optim"

-- Parse command-line options
local opt = lapp[[
   -g,--gpu                                 test on GPU
   -f,--full                                use the full test dataset
   -t,--threads       (default 4)           number of threads
]]

-- Fix seed
torch.manualSeed(1)

-- Threads
torch.setnumthreads(opt.threads)
print('Set nb of threads to ' .. torch.getnumthreads())

print("Setting up testing dataset...")
local feature_dim = 39  -- 13 MFCCs, 13 delta MFCCS, 13 delta-delta MFCCs
local context_frames = 10
local dataset={}
local features_file="/pool001/atitus/FastLID-DNN/data_prep/feats/features_test_labeled"
local dataset_size = 0
local lang2label = {outofset = 1, english = 2, german = 3, mandarin = 4}
local label2uttcount = torch.zeros(4)
local total_utterances = 1174

-- Only use full dataset if we say so
local max_utterances = 100
if opt.full then
    max_utterances = total_utterances
end
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
        label2uttcount[label] = label2uttcount[label] + 1

        -- Read in features into a tensor
        local feature_strs = string.sub(line, lang_j + 1)
        local features_tensor = torch.Tensor(feature_dim)
        if opt.gpu then
            features_tensor = features_tensor:cuda()
        end
        local feature_idx = 1
        for feature_str in string.gmatch(feature_strs, "[%-]?[0-9]*%.[0-9]*") do
            features_tensor[feature_idx] = tonumber(feature_str)
            feature_idx = feature_idx + 1
        end

        -- Add this to the dataset
        dataset[dataset_size + 1] = {features_tensor, label, current_utterance}
        dataset_size = dataset_size + 1
    end
end
function dataset:size() return dataset_size end
print("Done setting up dataset with " .. dataset_size .. " datapoints across " .. max_utterances .. " utterances.")

-- Load individual DNNs for ensemble
local english_dnn = "/pool001/atitus/FastLID-DNN/models/english_1k_1k_Adam_e489_b512"
print("Loading neural network for English " .. english_dnn .. "...")
english_model = torch.load(english_dnn)
print("Loaded neural network " .. english_dnn)
print(english_model)

local german_dnn = "/pool001/atitus/FastLID-DNN/models/german_1k_1k_Adam_e1000_b512"
print("Loading neural network for German " .. german_dnn .. "...")
german_model = torch.load(german_dnn)
print("Loaded neural network " .. german_dnn)
print(german_model)

local mandarin_dnn = "/pool001/atitus/FastLID-DNN/models/mandarin_1k_1k_Adam_e1000_b512"
print("Loading neural network for Mandarin Chinese " .. mandarin_dnn .. "...")
mandarin_model = torch.load(mandarin_dnn)
print("Loaded neural network " .. mandarin_dnn)
print(mandarin_model)

-- Set up DNNs for testing (i.e. deactivate Dropout)
english_model:evaluate()
german_model:evaluate()
mandarin_model:evaluate()

if opt.gpu then
    -- Convert our networks to CUDA-compatible version
    print("Convert networks to CUDA")
    english_model = english_model:cuda()
    german_model = german_model:cuda()
    mandarin_model = mandarin_model:cuda()
end

-- Set up confusion matrix
local labels = {1, 2, 3, 4}
local confusion = optim.ConfusionMatrix(labels)

local correct_frames = 0
local utterance_output_avgs = {}        -- averaged output probabilities
local utterance_frame_counts = {}       -- current count of probabilities (used for averaging)
local utterance_labels = {}             -- correct utterance-level label
local current_utterance = ""
local utterance_count = 0

local start_time = sys.clock()

local input = torch.zeros(feature_dim * (context_frames + 1))
local overall_output_probs = torch.zeros(#labels)
if opt.gpu then
    -- Convert our tensors to CUDA-compatible version
    print("Convert input and output tensors to CUDA")
    input = input:cuda()
    overall_output_probs = overall_output_probs:cuda()
end

-- Determined based on classifier convergence values
local ensemble_weights = {english = 1.0337727964, german = 0.9372224005, mandarin = 1.029004803}

print("Testing neural network ensemble...")
for i=1,dataset:size() do
    local data = dataset[i]
    local features_tensor = data[1]
    local label = data[2]
    local utt = data[3]

    -- Load current features
    input:zero()
    input[{ {1, feature_dim} }] = features_tensor

    -- Load context features, if any
    for context = 1, math.min(context_frames, i - 1) do
        local context_data = dataset[i - context]
        local context_features_tensor = context_data[1]
        local context_label = context_data[2] 
        local slice_begin = (context * feature_dim) + 1
        local slice_end = (context + 1) * feature_dim
        input[{ {slice_begin, slice_end} }] = context_features_tensor
    end

    -- Evaluate this frame
    -- TODO: run these in parallel
    local classification = lang2label["outofset"]       -- out-of-set, to start
    local current_confidence = 0.5                      -- random guess

    -- Convert log probabilities (from logsoftmax layer) to probabilities
    local english_output_logprobs = english_model:forward(input)
    local english_output_probs = torch.exp(english_output_logprobs)
    local english_confidence = english_output_probs[2] * ensemble_weights["english"]
    if english_confidence >= current_confidence then
        classification = lang2label["english"]
        current_confidence = english_confidence
    end
    
    local german_output_logprobs = german_model:forward(input)
    local german_output_probs = torch.exp(german_output_logprobs)
    local german_confidence = german_output_probs[2] * ensemble_weights["german"]
    if german_confidence >= current_confidence then
        classification = lang2label["german"]
        current_confidence = german_confidence
    end
   
    local mandarin_output_logprobs = mandarin_model:forward(input)
    local mandarin_output_probs = torch.exp(mandarin_output_logprobs)
    local mandarin_confidence = mandarin_output_probs[2] * ensemble_weights["mandarin"]
    if mandarin_confidence >= current_confidence then
        classification = lang2label["mandarin"]
        current_confidence = mandarin_confidence
    end

    -- Set output probs to be 100% our classification (since the sum, product, etc. doesn't make sense in this context)
    overall_output_probs:zero()
    overall_output_probs[{ {classification} }] = current_confidence

    -- Check if the ensemble classified correctly
    if classification == label then
        correct_frames = correct_frames + 1
    end
            
    -- Update confusion matrix
    confusion:add(overall_output_probs, label)

    -- Update utterance-level stats
    if utt ~= current_utterance then
        -- Create new entry
        utterance_count = utterance_count + 1
        utterance_output_avgs[utterance_count] = overall_output_probs
        utterance_labels[utterance_count] = label
        utterance_frame_counts[utterance_count] = 1
        current_utterance = utt
    else
        -- Update average
        local old_frame_count = utterance_frame_counts[utterance_count]
        local old_avg = utterance_output_avgs[utterance_count]
        local new_avg = torch.mul(torch.add(overall_output_probs, torch.mul(old_avg, old_frame_count)), 1.0 / (old_frame_count + 1))
        utterance_output_avgs[utterance_count] = new_avg
        utterance_frame_counts[utterance_count] = utterance_frame_counts[utterance_count] + 1
    end
end

-- Print time statistics for frame-level testing
local end_time = sys.clock()
local elapsed_time = end_time - start_time
local time_per_sample = elapsed_time / dataset:size()
print("================================")
print("Frame-Level Testing:")
print("  time to test 1 sample = " .. (time_per_sample * 1000) .. "ms")
print("  time to test all " .. dataset:size() .. " samples = " .. (elapsed_time * 1000) .. "ms")
print("================================")

-- Print confusion matrix and reset
print(confusion)
confusion:zero()

local start_time = sys.clock()

local correct_utterances = 0
for i=1,max_utterances do
    -- Test whole utterance
    local label = utterance_labels[i]
    local confidence, classification_tensor = torch.max(utterance_output_avgs[i], 1)
    local classification = classification_tensor[1]
    if classification == label then
        correct_utterances = correct_utterances + 1
    end
    
    -- Update confusion matrix
    confusion:add(utterance_output_avgs[i], label)
end

-- Print time statistics for utterance-level testing
local end_time = sys.clock()
local elapsed_time = end_time - start_time
local time_per_utterance = elapsed_time / max_utterances
print("================================")
print("Utterance-Level Testing:")
print("  time to test 1 utterance = " .. (time_per_utterance * 1000) .. "ms")
print("  time to test all " .. max_utterances .. " utterances = " .. (elapsed_time * 1000) .. "ms")
print("================================")

-- Print confusion matrix and reset
print(confusion)
confusion:zero()
print("Done testing neural network.")
