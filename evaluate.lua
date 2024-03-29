require "torch"
require "nn"
require "optim"

local lre03DatasetReader = require "lre03DatasetReader"

-- Parse command-line options
local opt = lapp[[
   -n,--network       (string)              reload pretrained network
   -g,--gpu                                 evaluate on GPU
   -t,--threads       (default 4)           number of threads
   --languages        (string)              languages being used, delimited by "_"
]]

if opt.gpu then
    require "cunn"
end

-- Fix seed
torch.manualSeed(1)

-- Threads
torch.setnumthreads(opt.threads)
print('Set nb of threads to ' .. torch.getnumthreads())

print("Setting up evaluation dataset...")
--local features_file="/pool001/atitus/FastLID-DNN/data_prep/feats/" .. opt.languages .. "_evaluate"
local features_file="/pool001/atitus/FastLID-DNN/data_prep/feats_all/" .. opt.languages .. "_evaluate"
--local lang2label = {outofset = 1, english = 2, german = 3, mandarin = 4}
local lang2label = {outofset = 1, vietnamese = 2, tamil = 3, spanish = 4, farsi = 5, korean = 6, japanese = 7, hindi = 8, french = 9, english = 10, german = 11, mandarin = 12, arabic = 13}

-- Balance data
local total_frames = 21746      -- Count for Korean, the minimum of all languages in set
--local total_frames = 26326      -- Count for German, the minimum in this label set
--local total_frames = 47907      -- Amount in Mandarin, the minimum of this language set
--local label2maxframes = torch.zeros(4)
local label2maxframes = torch.zeros(13)
for lang, label in pairs(lang2label) do
    label2maxframes[label] = total_frames
end

--label2maxframes[lang2label["outofset"]] = total_frames
--label2maxframes[lang2label["english"]] = total_frames
--label2maxframes[lang2label["german"]] = total_frames
--label2maxframes[lang2label["mandarin"]] = total_frames

print("Loading neural network " .. opt.network .. "...")
model = torch.load(opt.network)
print("Loaded neural network " .. opt.network)

-- Set up for evaluation (i.e. deactivate Dropout)
model:evaluate()
print("Using model:")
print(model)

if opt.gpu then
    -- Convert our network to CUDA-compatible version
    print("Convert network to CUDA")
    model = model:cuda()
end

-- Load the evaluation dataset
local feature_dim = 39  -- 13 MFCCs, 13 delta MFCCS, 13 delta-delta MFCCs
local context_frames = 20
--local max_utterances = 359      -- English, German, Mandarin
--local max_utterances = 267      -- English, German
--local max_utterances = 273      -- German, Mandarin
--local max_utterances = 486      -- English, Mandarin
local max_utterances = 895      -- All languages in set
local readCfg = {
    features_file = features_file,
    lang2label = lang2label,
    label2maxframes = label2maxframes,
    include_utts = true,
    gpu = opt.gpu
}
local dataset, label2framecount = lre03DatasetReader.read(readCfg)

-- Set up confusion matrix
--labels = {1, 2, 3, 4}
labels = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
local confusion = optim.ConfusionMatrix(labels)

print("Testing neural network...")
local correct_frames = 0
local utterance_output_avgs = {}        -- averaged output probabilities
local utterance_labels = {}             -- correct utterance-level label
local utterance_ids = {}                -- utterance IDs from NIST files
local utterance_frame_counts = {}       -- frames seen so far for given utterance
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

    -- Evaluate this frame
    local output_probs = model:forward(input)

    local confidence_tensor, classification_tensor = torch.max(output_probs, 1)
    local confidence = confidence_tensor[1]
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
        utterance_labels[utterance_count] = label
        utterance_frame_counts[utterance_count] = 1
        utterance_output_avgs[utterance_count] = output_probs
        utterance_ids[utterance_count] = utt
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

-- Print time statistics for frame-level evaluation
local end_time = sys.clock()
local elapsed_time = end_time - start_time
local time_per_sample = elapsed_time / dataset:size()
local fer = 1.0 - (correct_frames / dataset:size())
print("================================")
print("Frame-Level Testing:")
print("  time to evaluate 1 sample = " .. (time_per_sample * 1000) .. "ms")
print("  time to evaluate all " .. dataset:size() .. " samples = " .. (elapsed_time * 1000) .. "ms")
print("  FER: " .. fer)
print("================================")

-- Print confusion matrix and reset
print(confusion)
confusion:zero()

local start_time = sys.clock()

local correct_utterances = 0
for i=1,max_utterances do
    -- Test whole utterance
    local label = utterance_labels[i]
    local utt_id = utterance_ids[i]
    local confidence, classification_tensor = torch.max(utterance_output_avgs[i], 1)
    local classification = classification_tensor[1]
    if classification == label then
        correct_utterances = correct_utterances + 1
    else
        print("Incorrectly classified " .. label .. " utterance " .. utt_id .. " as " .. classification)
    end
    
    -- Update confusion matrix
    confusion:add(utterance_output_avgs[i], label)
end

-- Print time statistics for utterance-level evaluation
local end_time = sys.clock()
local elapsed_time = end_time - start_time
local time_per_utterance = elapsed_time / max_utterances
local uer = 1.0 - (correct_utterances / max_utterances)
print("================================")
print("Utterance-Level Testing:")
print("  time to evaluate 1 utterance = " .. (time_per_utterance * 1000) .. "ms")
print("  time to evaluate all " .. max_utterances .. " utterances = " .. (elapsed_time * 1000) .. "ms")
print("  UER: " .. uer)
print("================================")

-- Print confusion matrix and reset
print(confusion)
confusion:zero()
print("Done evaluating neural network.")
