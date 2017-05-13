require "torch"

-- "Class" for reading feature datasets from wav files
local featuresReader = {}

-- Read dataset from given features file
--   takes features_file as input: string indicating absolute path of features file
--   Returns dataset
function featuresReader.read(features_file)
    print("Reading dataset...")
    local dataset={}
    local dataset_size = 0
    local feature_dim = 39  -- 13 MFCCs, 13 delta MFCCS, 13 delta-delta MFCCs

    for line in io.lines(features_file) do
        -- Read in features into a tensor
        local feature_tensor = torch.zeros(feature_dim)
        local feature_idx = 1
        for feature_str in string.gmatch(line, "[%-]?[0-9]*%.[0-9]*") do
            feature_tensor[feature_idx] = tonumber(feature_str)
            feature_idx = feature_idx + 1
        end

        -- Add this to the dataset
        dataset_size = dataset_size + 1
        dataset[dataset_size] = feature_tensor
    end
    function dataset:size() return dataset_size end
    print("Done setting up dataset with " .. dataset:size() .. " frames")
    return dataset
end

return lre03DatasetReader
