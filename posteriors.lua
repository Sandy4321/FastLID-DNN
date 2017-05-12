require "torch"
require "nn"

local lre03DatasetReader = require "lre03DatasetReader"

-- Parse command-line options
local opt = lapp[[
   -n,--network       (string)              reload pretrained network
   -g,--gpu                                 evaluate on GPU
   -t,--threads       (default 4)           number of threads
   --languages         (string)             languages being used, delimited by "_"
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
local features_file="/pool001/atitus/FastLID-DNN/data_prep/feats/" .. opt.languages .. "_evaluate"
local lang2label = {outofset = 1, english = 2, german = 3, mandarin = 1}

-- Balance data
local total_frames = 26326      -- Amount in German, the minimum of this language set
local label2maxframes = torch.zeros(4)
label2maxframes[lang2label["outofset"]] = total_frames
label2maxframes[lang2label["english"]] = total_frames
label2maxframes[lang2label["german"]] = total_frames
label2maxframes[lang2label["mandarin"]] = total_frames

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
local max_utterances = 359
local readCfg = {
    features_file = features_file,
    lang2label = lang2label,
    label2maxframes = label2maxframes,
    include_utts = true,
    gpu = opt.gpu
}
local dataset, label2framecount = lre03DatasetReader.read(readCfg)

print("Testing neural network posteriors...")
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

    -- Update utterance-level stats
    if utt ~= current_utterance then
        -- Create new entry
        current_utterance = utt
        print("UTTERANCE " .. current_utterance .. " WITH LABEL " .. label)
    end
        
    -- Print posteriors to our log
    --print("Frame " .. i .. " posteriors:" .. output_probs[lang2label["outofset"]] .. "," .. output_probs[lang2label["english"]] .. "," .. output_probs[lang2label["german"]] .. "," .. output_probs[lang2label["mandarin"]])
    print("Frame " .. i .. " posteriors:" .. output_probs[lang2label["outofset"]] .. "," .. output_probs[lang2label["english"]] .. "," .. output_probs[lang2label["german"]])
end
