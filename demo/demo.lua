require "torch"
require "nn"

local featuresReader = require "featuresReader"

-- Parse command-line options
local opt = lapp[[
   -n,--network       (string)              reload pretrained network
   -f,--featuresFile  (string)              filename to load features from
   -t,--threads       (default 4)           number of threads
   --numLanguages     (default 4)           number of language classes, including out-of-set
]]

-- Fix seed
torch.manualSeed(1)

-- Threads
torch.setnumthreads(opt.threads)
print('Set nb of threads to ' .. torch.getnumthreads())

print("Loading neural network " .. opt.network .. "...")
model = torch.load(opt.network)
print("Loaded neural network " .. opt.network)

-- Set up for evaluation (i.e. deactivate Dropout if necessary)
model:evaluate()
print("Using model:")
print(model)

-- Load the features dataset
print("Setting up features...")
local feature_dim = 39  -- 13 MFCCs, 13 delta MFCCS, 13 delta-delta MFCCs
local context_frames = 20
local features_file = opt.featuresFile
local dataset = featuresReader.read(features_file)

local utterance_output_avg = torch.zeros(4)     -- averaged output probabilities
local start_time = sys.clock()
local input = torch.zeros(feature_dim * (context_frames + 1))

local frame_count = 0

for i=1,dataset:size() do
    local features_tensor = dataset[i]

    -- Load current features
    input[{ {1, feature_dim} }] = features_tensor

    -- Load context features, if any
    for context = 1, math.min(context_frames, i - 1) do
        local context_data = dataset[i - context]
        local context_features_tensor = context_data[1]

        local slice_begin = (context * feature_dim) + 1
        local slice_end = (context+1)*feature_dim
        input[{ {slice_begin, slice_end} }] = context_features_tensor
    end

    -- Evaluate this frame
    local output_probs = model:forward(input)
    
    -- Update average
    local old_avg = utterance_output_avg
    utterance_output_avg = torch.mul(torch.add(output_probs, torch.mul(old_avg, frame_count)), 1.0 / (frame_count + 1))
    frame_count = frame_count + 1
end

local end_time = sys.clock()
local elapsed_time = end_time - start_time

-- Make an utterance-level classification
local langs = {"Out-of-Set", "English", "German", "Mandarin Chinese"}
local confidence_tensor, classification_tensor = torch.max(utterance_output_avg, 1)
local log_confidence = confidence_tensor[1]
local confidence = math.exp(log_confidence)
local classification = langs[tonumber(classification_tensor[1])]

-- Print time statistics for frame-level evaluation
print("================================")
print("Language is " .. classification .. " with confidence " .. (confidence * 100) .. "%")
print("time to evaluate = " .. (elapsed_time * 1000) .. "ms")
print("================================")
