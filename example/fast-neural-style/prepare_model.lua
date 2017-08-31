require 'torch'
require 'nn'

require 'fast-neural-style.fast_neural_style.ShaveImage'
require 'fast-neural-style.fast_neural_style.TotalVariation'

torch.setdefaulttensortype('torch.FloatTensor')

local function replaceModule(x, name, create_fn)
  if not x.modules then
    return
  end
  for i = 1,#x.modules do
    m = x.modules[i]
    if m.__typename == name then
      x.modules[i] = create_fn(m)
    end
    replaceModule(m, name, create_fn)
  end
end

local function main()
  local cmd = torch.CmdLine()
  cmd:option('-input', '')
  cmd:option('-output', '')
  local opt = cmd:parse(arg)
  local model = torch.load(opt.input).model

  -- Replace nn.ShaveImage with crop using SpatialZeroPadding with negative offsets
  replaceModule(model, 'nn.ShaveImage', function(m)
    local size = m.size
    return nn.SpatialZeroPadding(-size, -size, -size, -size)
  end)

  -- Remove last TotalVariation layer
  if model.modules[#model.modules].__typename == 'nn.TotalVariation' then
    model.modules[#model.modules] = nil
  end

  -- Save prepared model
  torch.save(opt.output, model)
end

main()
