require "torch"

-- "Class" for reading LRE 2003 datasets
local lre03DatasetReader = {}

-- Read dataset from given features file
--   Takes config file as input
--      features_file: string indicating absolute path of features file
--      lang2label: table mapping language strings to number labels
--      label2maxframes: tensor indicating the maximum number of frames allowed for a given label
--      include_utts: boolean indicating whether to add utterance ID as third dim in dataset
--      gpu: boolean indicating whether data will be used on GPU
--   Returns dataset, as well as table mapping labels to utterance counts
function lre03DatasetReader.read(cfg)
    print("Reading dataset...")
    local dataset={}
    local dataset_size = 0

    if cfg.gpu then
        require "cunn"
    end

    --local languages = {"outofset", "english", "german", "mandarin"}
    --local label2framecount = torch.zeros(4)      -- Assumes max language set size is 3 (see paper)
    --local label2uttcount = torch.zeros(4)      -- Assumes max language set size is 3 (see paper)
    local languages = {"outofset", "vietnamese", "tamil", "spanish", "farsi", "korean", "japanese", "hindi", "french", "english", "german", "mandarin", "arabic"}
    local label2framecount = torch.zeros(13)      -- Full language set
    local label2uttcount = torch.zeros(13)      -- Full language set

    local feature_dim = 39  -- 13 MFCCs, 13 delta MFCCS, 13 delta-delta MFCCs

    local current_utterance=""
    local utterances_used = 0
    for line in io.lines(cfg.features_file) do
        -- Find utterance ID
        local utt = string.sub(line, 1, 4)

        -- Find language label
        local lang_i, lang_j = string.find(line, "[a-z]+ ", 5)
        local lang = string.sub(line, lang_i, lang_j - 1)   -- Cut off trailing whitespace
        local label = cfg.lang2label[lang]

        -- Check if we should use this utterance
        if utt ~= current_utterance then
            if label2framecount[label] < cfg.label2maxframes[label] then
                current_utterance = utt
                utterances_used = utterances_used + 1
                label2uttcount[label] = label2uttcount[label] + 1
            end
        end

        -- Check if we should bail
        local bail = true
        --for lang_idx = 1, 4 do
        for lang_idx = 1, 13 do
            local current_label = cfg.lang2label[languages[lang_idx]]
            if label2framecount[current_label] < cfg.label2maxframes[current_label] then
                -- Still have room for frames in this label
                bail = false
                break
            end
        end
        if bail then
            -- Undo the utterance we just added
            utterances_used = utterances_used - 1
            label2uttcount[label] = label2uttcount[label] - 1
            break
        end

        if utt == current_utterance then
            -- Add the utterance
            label2framecount[label] = label2framecount[label] + 1

            -- Read in features into a tensor
            local feature_strs = string.sub(line, lang_j + 1)
            local feature_tensor = torch.zeros(feature_dim)
            if cfg.gpu then
                feature_tensor = feature_tensor:cuda()
            end
            local feature_idx = 1
            for feature_str in string.gmatch(feature_strs, "[%-]?[0-9]*%.[0-9]*") do
                feature_tensor[feature_idx] = tonumber(feature_str)
                feature_idx = feature_idx + 1
            end

            -- Add this to the dataset
            dataset_size = dataset_size + 1
            if cfg.include_utts then
                dataset[dataset_size] = {feature_tensor, label, utt}
            else
                dataset[dataset_size] = {feature_tensor, label}
            end
        end
    end
    function dataset:size() return dataset_size end
    print("Done setting up dataset with " .. dataset:size() .. " datapoints across " .. utterances_used .. " utterances.")
    print("Utterances per language:")
    print(label2uttcount)
    return dataset, label2framecount
end

return lre03DatasetReader
