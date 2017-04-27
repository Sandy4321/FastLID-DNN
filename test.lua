require "torch"
require "nn"
require "optim"

local lre03DatasetReader = require "lre03DatasetReader"

-- Parse command-line options
local opt = lapp[[
   -n,--network       (string)              reload pretrained network
   -g,--gpu                                 test on GPU
   --noOOS                                  do not train for Out-of-Set utterances
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

print("Setting up testing dataset...")
local features_file="/pool001/atitus/FastLID-DNN/data_prep/feats_3/features_test_labeled"
local lang2label = {outofset = 1, english = 2, german = 3, mandarin = 4}

-- Force all data to be used
local total_frames = 335583
local label2maxframes = torch.zeros(4)
label2maxframes[lang2label["outofset"]] = total_frames
if opt.noOOS then
    print("No out-of-set languages being used")
    label2maxframes[lang2label["outofset"]] = 0
end
label2maxframes[lang2label["english"]] = total_frames
label2maxframes[lang2label["german"]] = total_frames
label2maxframes[lang2label["mandarin"]] = total_frames

-- Load the testing dataset
local feature_dim = 39  -- 13 MFCCs, 13 delta MFCCS, 13 delta-delta MFCCs
local context_frames = 10
local max_utterances = 1174
local readCfg = {
    features_file = features_file,
    lang2label = lang2label,
    label2maxframes = label2maxframes,
    include_utts = true,
    gpu = opt.gpu
}
local dataset, label2uttcount = lre03DatasetReader.read(readCfg)

print("Loading neural network " .. opt.network .. "...")
model = torch.load(opt.network)
print("Loaded neural network " .. opt.network)

-- Set up for testing (i.e. deactivate Dropout)
model:evaluate()
print("Using model:")
print(model)

if opt.gpu then
    -- Convert our network to CUDA-compatible version
    print("Convert network to CUDA")
    model = model:cuda()
end

-- Set up confusion matrix
if opt.noOOS then
    labels = {2, 3, 4}
else
    labels = {1, 2, 3, 4}
end
local confusion = optim.ConfusionMatrix(labels)

print("Testing neural network...")
local correct_frames = 0
local utterance_output_avgs = {}        -- averaged output probabilities
local utterance_frame_counts = {}       -- current count of probabilities (used for averaging)
local utterance_labels = {}             -- correct utterance-level label
local utterance_count = 0

local start_time = sys.clock()

local input = torch.zeros(feature_dim * (context_frames + 1))
if opt.gpu then
    -- Convert our input tensor to CUDA-compatible version
    print("Convert input tensor to CUDA")
    input = input:cuda()
end
for i=1,dataset:size() do
    local data = dataset[i]
    local features_tensor = data[1]
    local label = data[2]
    if opt.noOOS then
        label = label - 1   -- No OOS - shift over labels
    end
    local utt = data[3]

    -- Load current features
    input:zero()
    input[{ {1, feature_dim} }] = features_tensor

    -- Load context features, if any
    for context = 1, math.min(context_frames, i - 1) do
        local context_data = dataset[i - context]
        local context_features_tensor = context_data[1]
        local context_utt = context_data[3]

        -- Don't let another utterance spill over into this one!
        if context_utt == utt then
            local slice_begin = (context * feature_dim) + 1
            local slice_end = (context+1)*feature_dim
            input[{ {slice_begin, slice_end} }] = context_features_tensor
        end
    end

    -- Evaluate this frame and convert negative log likelihoods to probabilities
    local output_nlls = model:forward(input)
    local output_probs = torch.exp(output_nlls)
    local confidence, classification_tensor = torch.max(output_probs, 1)
    local classification = classification_tensor[1]
    if classification == label then
        correct_frames = correct_frames + 1
    end
            
    -- Update confusion matrix
    confusion:add(output_probs, label)

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

-- Print time statistics for frame-level testing
local end_time = sys.clock()
local elapsed_time = end_time - start_time
local time_per_sample = elapsed_time / dataset:size()
print("================================")
print("Frame-Level Testing:")
print("  time to test 1 sample = " .. (time_per_sample * 1000) .. "ms")
print("  time to test all " .. dataset:size() .. " samples = " .. (elapsed_time * 1000) .. "ms")
print("  FER: " .. (correct_frames / dataset:size()))
print("================================")

-- Print confusion matrix and reset
print(confusion)
confusion:zero()

local start_time = sys.clock()

local correct_utterances = 0
for i=1,max_utterances do
    -- Test whole utterance
    local label = utterance_labels[i]
    print("Label is " .. label .. ", probs are:")
    print(utterance_output_avgs[i])
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
print("  UER: " .. (correct_utterances / max_utterances))
print("================================")

-- Print confusion matrix and reset
print(confusion)
confusion:zero()
print("Done testing neural network.")
