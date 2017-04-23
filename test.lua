require "torch"
require "nn"
require "cunn"

print("Setting up test dataset...")
local feature_dim = 39  -- 13 MFCCs, 13 delta MFCCS, 13 delta-delta MFCCs
local dataset={}
local features_file="/pool001/atitus/FastLID-DNN/data_prep/feats/features_test_labeled"
local dataset_size=0
local labels = {outofset = 1, english = 2, german = 3, mandarin = 4}
local total_utterances=1174
local max_utterances=1174
local current_utterance_count=0
print("Using a total of " .. max_utterances .. " utterances of dataset")

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
        dataset[dataset_size + 1] = {feature_tensor, label, current_utterance}
        dataset_size = dataset_size + 1
    end
end
function dataset:size() return dataset_size end
print("Done setting up dataset with " .. dataset_size .. " datapoints across " .. max_utterances .. " utterances.")

print("Loading neural network...")
local net_filename = "/pool001/atitus/FastLID-DNN/models/1k_1k_truncated_lr001"
mlp = torch.load(net_filename)
print("Done loading neural network.")

print("Testing neural network...")
local correct_frames = 0
local utterance_output_avgs = {}        -- averaged output probabilities
local utterance_frame_counts = {}       -- current count of probabilities (used for averaging)
local utterance_labels = {}             -- correct utterance-level label
local current_utterance = ""
local utterance_count = 0

for i=1,dataset_size do
    local data = dataset[i]
    local feature_tensor = data[1]
    local label = data[2]
    local utt = data[3]

    -- Evaluate this frame
    local output_probs = mlp:forward(feature_tensor)
    local confidence, classification_tensor = torch.max(output_probs, 1)
    local classification = classification_tensor[1]
    if classification == label then
        correct_frames = correct_frames + 1
    end

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

local correct_utterances = 0
for i=1,max_utterances do
    local correct_label = utterance_labels[i]
    local confidence, classification_tensor = torch.max(utterance_output_avgs[i], 1)
    local classification = classification_tensor[1]
    if classification == correct_label then
        correct_utterances = correct_utterances + 1
    end
    print("Utterance " .. i .. " label " .. correct_label .. ", got " .. classification)
end

print("Done testing neural network.")

print("=================================")
print("Test accuracy per utterance: " .. (correct_utterances / max_utterances))
print("Test accuracy per frame: " .. (correct_frames / dataset_size))
print("=================================")
