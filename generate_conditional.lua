require 'torch'
require 'nn'
require 'image'
require 'cunn'
require 'cudnn'

opt = {
  model_base = 'checkpoints/condgolf7/',
  model = 'iter170000',
  dataset = 'video2',   -- indicates what dataset load to use (in data.lua)
  batchSize = 32,
  loadSize = 128,       -- when loading images, resize first to this size
  fineSize = 64,
  frameSize = 32,
  nThreads = 32,        -- how many threads to pre-fetch data
  randomize = 1,        -- whether to shuffle the data file or not
  gpu = 1,
  cudnn = 1,
  mean = {0,0,0},
  data_root = '/home/mschoi/workspace/videogan/data/golf-frames-stable/',
  data_list = '/home/mschoi/workspace/videogan/data/golf.txt',
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

torch.manualSeed(0)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- if using GPU, select indicated one
cutorch.setDevice(opt.gpu)

net  = torch.load(opt.model_base .. opt.model .. '_net.t7') 
net:evaluate()
net:cuda()
net = cudnn.convert(net, cudnn)

print('Generator:')
print(net)

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

local test_input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize):cuda()
local input_data = data:getBatch()
test_input:copy(input_data:select(3,1))

local gen = net:forward(test_input)
local video = net.modules[3].output[1]:float()
local mask = net.modules[3].output[2]:float()
local static = net.modules[3].output[3]:float()
local mask = mask:repeatTensor(1,3,1,1,1)

function WriteGif(filename, movie)
  for fr=1,movie:size(3) do
    image.save(filename .. '.' .. string.format('%08d', fr) .. '.png', image.toDisplayTensor(movie:select(3,fr)))
  end
  cmd = "ffmpeg -f image2 -i " .. filename .. ".%08d.png -y " .. filename
  print('==> ' .. cmd)
  sys.execute(cmd)
  for fr=1,movie:size(3) do
    os.remove(filename .. '.' .. string.format('%08d', fr) .. '.png')
  end
end

paths.mkdir('vis/')
local p = 'vis/' .. opt.model
paths.mkdir(p)
WriteGif(p .. '/gen.gif', gen) 
WriteGif(p .. '/video.gif', video) 
WriteGif(p .. '/videomask.gif', torch.cmul(video, mask))
WriteGif(p .. '/mask.gif', mask)
image.save(p .. '/static.jpg', image.toDisplayTensor(static))
