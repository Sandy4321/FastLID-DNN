require "torch"
require "nn"

print("Setting up training dataset...")
local feature_dim = 39  -- 13 MFCCs, 13 delta MFCCS, 13 delta-delta MFCCs
local dataset={}
local features_file="/pool001/atitus/FastLID-DNN/data_prep/feats/features_train_labeled"
local dataset_size=0
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
    dataset[dataset_size + 1] = {lang, feature_tensor}
    dataset_size = dataset_size + 1
end
function dataset:size() return dataset_size end
print("Done setting up dataset.")
