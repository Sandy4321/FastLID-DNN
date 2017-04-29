require "torch"

local lre03DatasetReader = require "lre03DatasetReader"

local lang2label = {outofset = 1, english = 2, german = 3, mandarin = 4}

-- Balance the data
local validate_file="/pool001/atitus/FastLID-DNN/data_prep/feats/features_validate_labeled"
local min_frames = 24841   -- Count for German, the minimum in this label set
local label2maxframes = torch.zeros(4)
label2maxframes[lang2label["outofset"]] = min_frames
label2maxframes[lang2label["english"]] = min_frames
label2maxframes[lang2label["german"]] = min_frames
label2maxframes[lang2label["mandarin"]] = min_frames
label2maxframes:floor()

-- Load the validation dataset
local readCfg = {
    features_file = validate_file,
    lang2label = lang2label,
    label2maxframes = label2maxframes,
    include_utts = true,
    gpu = true
}
print("Loading validation dataset")
local dataset, label2framecount = lre03DatasetReader.read(readCfg)
print(label2framecount)

-- Balance the data
local evaluate_file="/pool001/atitus/FastLID-DNN/data_prep/feats/features_evaluate_labeled"
local min_frames = 23507   -- Count for German, the minimum in this label set
local label2maxframes = torch.zeros(4)
label2maxframes[lang2label["outofset"]] = min_frames
label2maxframes[lang2label["english"]] = min_frames
label2maxframes[lang2label["german"]] = min_frames
label2maxframes[lang2label["mandarin"]] = min_frames
label2maxframes:floor()

-- Load the evaluation dataset
local readCfg = {
    features_file = evaluate_file,
    lang2label = lang2label,
    label2maxframes = label2maxframes,
    include_utts = true,
    gpu = true
}
print("Loading evaluation dataset")
local dataset, label2framecount = lre03DatasetReader.read(readCfg)
print(label2framecount)

-- Balance the data
local train_file="/pool001/atitus/FastLID-DNN/data_prep/feats/features_train_labeled"
local min_frames = 342643   -- Count for German, the minimum in this label set
local label2maxframes = torch.zeros(4)
label2maxframes[lang2label["outofset"]] = min_frames
label2maxframes[lang2label["english"]] = min_frames
label2maxframes[lang2label["german"]] = min_frames
label2maxframes[lang2label["mandarin"]] = min_frames
label2maxframes:floor()

-- Load the training dataset
local readCfg = {
    features_file = train_file,
    lang2label = lang2label,
    label2maxframes = label2maxframes,
    include_utts = true,
    gpu = true
}
print("Loading training dataset")
local dataset, label2framecount = lre03DatasetReader.read(readCfg)
print(label2framecount)
