require "torch"
require "nn"

print("Setting up training dataset...")
local feature_dim = 39  -- 13 MFCCs, 13 delta MFCCS, 13 delta-delta MFCCs
local dataset={}
local features_file="/pool001/atitus/FastLID-DNN/data_prep/feats/features_train_labeled"
local dataset_size=0
for line in io.lines(features_file) do
    local utt = string.sub(line, 1, 4)
    local  
end
function dataset:size() return dataset_size end
print("Done setting up dataset.")
