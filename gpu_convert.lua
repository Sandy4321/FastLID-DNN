require "torch"
require "nn"

-- Parse command-line options
local opt = lapp[[
   -n,--network       (string)              load pretrained network
   -t,--threads       (default 4)           number of threads
]]

require "cunn"

print("Loading neural network " .. opt.network .. "...")
model = torch.load(opt.network)
print("Loaded neural network " .. opt.network)

-- Set up for evaluation (i.e. deactivate Dropout)
model:evaluate()
print("Loaded model:")
print(model)

-- Convert to non-GPU version
print("Converting to non-GPU version and saving...")
model:double()
torch.save(opt.network .. "_nongpu", model)
print("Saved model")
