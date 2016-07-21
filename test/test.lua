require 'cudnn'
require 'cunn'

local cudnntest = torch.TestSuite()
local nloop = 1
local times = {}
local mytester
local jac = nn.Jacobian


local testparams_half = {
   test_type = 'torch.CudaHalfTensor',
   precision_forward = 2e-1,
   precision_backward = 4,
   precision_jac = 1e-3,
   precision_io = 1e-1,
}

local testparams_float = {
   test_type = 'torch.CudaTensor',
   precision_forward = 1e-4,
   precision_backward = 1e-2,
   precision_jac = 1e-3,
   precision_io = 1e-5,
}

-- TODO: find out why the errors are so huge
local testparams_double = {
   test_type = 'torch.CudaDoubleTensor',
   precision_forward = 1e+4,
   precision_backward = 1e+4,
   precision_jac = 1e-3,
   precision_io = 1e-5,
}

local testparams = testparams_half

local function cast(input)
   return input:type(testparams.test_type)
end

-- workarounds
function torch.CudaHalfTensor:abs()
   return self:cuda():abs():cudaHalf()
end

function torch.CudaDoubleTensor:abs()
   return self:cuda():abs():cudaDouble()
end

function torch.CudaHalfTensor:mean()
   return self:cuda():mean()
end

function torch.CudaDoubleTensor:mean()
   return self:cuda():mean()
end

function cudnntest.SpatialConvolution_forward_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local input = torch.randn(bs,from,inj,ini):cuda()
   local sconv = nn.SpatialConvolution(from,to,ki,kj,si,sj):cuda()
   local gconv = cast(cudnn.SpatialConvolution(from,to,ki,kj,si,sj)):fastest()
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)

   local function test(sconv, gconv)
     local groundtruth = sconv:forward(input)
     cutorch.synchronize()
     local rescuda = gconv:forward(cast(input))
     cutorch.synchronize()
     local error = rescuda:float() - groundtruth:float()
     mytester:assertlt(error:abs():max(), testparams.precision_forward, 'error on state (forward) ')

     -- IO
     local ferr,berr = jac.testIO(gconv, cast(input))
     mytester:assertlt(ferr, testparams.precision_io, torch.typename(gconv) .. ' - i/o forward err ')
     mytester:assertlt(berr, testparams.precision_io, torch.typename(gconv) .. ' - i/o backward err ')
   end

   test(sconv, gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv), 'cudnn.SpatialConvolution', 'conversion type check')
   test(sconv, gconv)
end


function cudnntest.SpatialConvolution_backward_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local scale = math.random()

   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,to,outj,outi):cuda()
   local sconv = nn.SpatialConvolution(from,to,ki,kj,si,sj):cuda()
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput, scale)
   cutorch.synchronize()
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias

   local gconv = cast(cudnn.SpatialConvolution(from,to,ki,kj,si,sj)):fastest()
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   gconv:forward(cast(input))

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   local function test(sconv, gconv)
     gconv:forward(cast(input))
     gconv:zeroGradParameters()
     local rescuda = gconv:backward(cast(input), cast(gradOutput), scale)
     cutorch.synchronize()
     local weightcuda = gconv.gradWeight
     local biascuda = gconv.gradBias

     local error = rescuda:float() - groundgrad:float()
     local werror = weightcuda:float() - groundweight:float()
     local berror = biascuda:float() - groundbias:float()

     mytester:assertlt(error:abs():max(), testparams.precision_backward, 'error on state (backward) ')
     mytester:assertlt(werror:abs():max(), testparams.precision_backward, 'error on weight (backward) ')
     mytester:assertlt(berror:abs():max(), testparams.precision_backward, 'error on bias (backward) ')
   end

   test(sconv, gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv), 'cudnn.SpatialConvolution', 'conversion type check')
   test(sconv, gconv)
end

function cudnntest.SpatialConvolution_forward_single()
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local input = torch.randn(from,inj,ini):cuda()
   local sconv = nn.SpatialConvolution(from,to,ki,kj,si,sj):cuda()
   local gconv = cast(cudnn.SpatialConvolution(from,to,ki,kj,si,sj))
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)

   local function test(sconv, gconv)
     local groundtruth = sconv:forward(input)
     cutorch.synchronize()
     local rescuda = gconv:forward(cast(input))
     cutorch.synchronize()
     mytester:asserteq(rescuda:dim(), 3, 'error in dimension')
     local error = rescuda:float() - groundtruth:float()
     mytester:assertlt(error:abs():max(), testparams.precision_forward,
                       'error on state (forward) ')

     -- IO
     local ferr,berr = jac.testIO(gconv, cast(input))
     mytester:assertlt(ferr, testparams.precision_io, torch.typename(gconv) .. ' - i/o forward err ')
     mytester:assertlt(berr, testparams.precision_io, torch.typename(gconv) .. ' - i/o backward err ')
   end

   test(sconv, gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv), 'cudnn.SpatialConvolution', 'conversion type check')
   test(sconv, gconv)
end


function cudnntest.SpatialConvolution_backward_single()
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local input = torch.randn(from,inj,ini):cuda()
   local gradOutput = torch.randn(to,outj,outi):cuda()
   local sconv = nn.SpatialConvolution(from,to,ki,kj,si,sj):cuda()
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias

   local gconv = cast(cudnn.SpatialConvolution(from,to,ki,kj,si,sj))
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   local function test(sconv, gconv)
     gconv:forward(cast(input))

     -- serialize and deserialize
     torch.save('modelTemp.t7', gconv)
     gconv = torch.load('modelTemp.t7')

     gconv:forward(cast(input))
     gconv:zeroGradParameters()
     local rescuda = gconv:backward(cast(input), cast(gradOutput))
     cutorch.synchronize()
     mytester:asserteq(rescuda:dim(), 3, 'error in dimension')
     local weightcuda = gconv.gradWeight
     local biascuda = gconv.gradBias

     local error = rescuda:float() - groundgrad:float()
     local werror = weightcuda:float() - groundweight:float()
     local berror = biascuda:float() - groundbias:float()

     mytester:assertlt(error:abs():max(), testparams.precision_backward,
                       'error on state (backward) ')
     mytester:assertlt(werror:abs():max(), testparams.precision_backward,
                       'error on weight (backward) ')
     mytester:assertlt(berror:abs():max(), testparams.precision_backward,
                       'error on bias (backward) ')
  end

  test(sconv, gconv)
  local gconv = cast(cudnn.convert(sconv, cudnn))
  mytester:asserteq(torch.typename(gconv), 'cudnn.SpatialConvolution', 'conversion type check')
  test(sconv, gconv)
end

function cudnntest.SpatialFullConvolution_forward_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local ini = math.random(1,64)
   local inj = math.random(1,64)
   local outi = (ini-1)*si+ki
   local outj = (inj-1)*sj+kj

   local input = torch.randn(bs,from,inj,ini):cuda()
   local sconv = nn.SpatialFullConvolution(from,to,ki,kj,si,sj):cuda()
   local gconv = cast(cudnn.SpatialFullConvolution(from,to,ki,kj,si,sj)):fastest()
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)

   local function test(sconv, gconv)
     local groundtruth = sconv:forward(input)
     cutorch.synchronize()
     local rescuda = gconv:forward(cast(input))
     cutorch.synchronize()
     local error = rescuda:float() - groundtruth:float()
     mytester:assertlt(error:abs():max(), testparams.precision_forward, 'error on state (forward) ')

     -- IO
     local ferr,berr = jac.testIO(gconv, cast(input))
     mytester:assertlt(ferr, testparams.precision_io, torch.typename(gconv) .. ' - i/o forward err ')
     mytester:assertlt(berr, testparams.precision_io, torch.typename(gconv) .. ' - i/o backward err ')
   end

   test(sconv, gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv), 'cudnn.SpatialFullConvolution', 'conversion type check')
   test(sconv, gconv)
end

function cudnntest.SpatialFullConvolution_backward_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local ini = math.random(1,64)
   local inj = math.random(1,64)
   local outi = (ini-1)*si+ki
   local outj = (inj-1)*sj+kj
   local scale = math.random()

   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,to,outj,outi):cuda()
   local sconv = nn.SpatialFullConvolution(from,to,ki,kj,si,sj):cuda()
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput, scale)
   cutorch.synchronize()
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias

   local gconv = cast(cudnn.SpatialFullConvolution(from,to,ki,kj,si,sj):cuda():fastest())
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   gconv:forward(cast(input))

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   local function test(sconv, gconv)
     gconv:forward(cast(input))
     gconv:zeroGradParameters()
     local rescuda = gconv:backward(cast(input), cast(gradOutput), scale)
     cutorch.synchronize()
     local weightcuda = gconv.gradWeight
     local biascuda = gconv.gradBias

     local error = rescuda:float() - groundgrad:float()
     local werror = weightcuda:float() - groundweight:float()
     local berror = biascuda:float() - groundbias:float()

     mytester:assertlt(error:abs():max(), testparams.precision_backward, 'error on state (backward) ')
     mytester:assertlt(werror:abs():max(), testparams.precision_backward, 'error on weight (backward) ')
     mytester:assertlt(berror:abs():max(), testparams.precision_backward, 'error on bias (backward) ')
   end

   test(sconv, gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv), 'cudnn.SpatialFullConvolution', 'conversion type check')
   test(sconv, gconv)
end


function cudnntest.TemporalConvolution_batch()
   local bs = math.random(1,32)
   local inputFrameSize = math.random(1,64)
   local outputFrameSize = math.random(1,64)
   local ki = math.random(1,15)
   local si = math.random(1,ki)
   local outi = math.random(1,15)
   local ini = (outi-1)*si+ki
   local scale = math.random()

   local input = torch.randn(bs,ini,inputFrameSize):cuda()
   local gradOutput = torch.randn(bs,outi,outputFrameSize):cuda()
   local sconv = nn.TemporalConvolution(inputFrameSize,outputFrameSize, ki, si):cuda()
   local groundForward = sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput, scale)
   cutorch.synchronize()
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias

   local gconv = cast(cudnn.TemporalConvolution(inputFrameSize,outputFrameSize, ki, si):cuda():fastest())
   gconv.weight:copy(sconv.weight:view(gconv.weight:size()))
   gconv.bias:copy(sconv.bias)
   gconv:forward(cast(input))

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   local cudaForward = gconv:forward(cast(input))
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(cast(input), cast(gradOutput), scale)
   cutorch.synchronize()
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias

   local ferror = cudaForward:float() - groundForward:float()
   local error = rescuda:float() - groundgrad:float()
   local werror = weightcuda:float() - groundweight:float()
   local berror = biascuda:float() - groundbias:float()
   mytester:assertlt(ferror:abs():max(), testparams.precision_forward, 'error on forward  ')
   mytester:assertlt(error:abs():max(), testparams.precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), testparams.precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), testparams.precision_backward, 'error on bias (backward) ')
end

function cudnntest.TemporalConvolution_padding_batch()
   local bs = math.random(1,32)
   local inputFrameSize = math.random(1,64)
   local outputFrameSize = math.random(1,64)
   local ki = math.random(2,15)
   local pad_h = math.floor(ki/2)
   local si = math.random(1,ki)
   local outi = math.random(2,15)
   local ini = (outi-1)*si+ki
   local scale = math.random()

   local inputpadded = torch.randn(bs,ini,inputFrameSize):cuda()
   for i=1,pad_h do
      inputpadded:narrow(2,i,1):fill(0)
      inputpadded:narrow(2,ini-i+1,1):fill(0)
   end
   local input = torch.Tensor(bs,ini - 2 * pad_h, inputFrameSize):cuda()
   input:copy(inputpadded:narrow(2, pad_h+1, ini - 2 * pad_h))
   local gradOutput = torch.randn(bs,outi,outputFrameSize):cuda()
   local sconv = nn.TemporalConvolution(inputFrameSize,outputFrameSize, ki, si):cuda()
   local groundForward = sconv:forward(inputpadded)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(inputpadded, gradOutput, scale)
   cutorch.synchronize()
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias

   local gconv = cast(cudnn.TemporalConvolution(inputFrameSize,outputFrameSize, ki, si,pad_h):cuda():fastest())
   gconv.weight:copy(sconv.weight:view(gconv.weight:size()))
   gconv.bias:copy(sconv.bias)
   gconv:forward(cast(input))

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   local cudaForward = gconv:forward(cast(input))
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(cast(input), cast(gradOutput), scale)
   cutorch.synchronize()
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias

   local ferror = cudaForward:float() - groundForward:float()
   groundgrad = groundgrad:narrow(2, pad_h + 1, ini - 2 * pad_h)
   local error = rescuda:float() - groundgrad:float()
   local werror = weightcuda:float() - groundweight:float()
   local berror = biascuda:float() - groundbias:float()
   mytester:assertlt(ferror:abs():max(), testparams.precision_forward, 'error on forward  ')
   mytester:assertlt(error:abs():max(), testparams.precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), testparams.precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), testparams.precision_backward, 'error on bias (backward) ')
end


function cudnntest.TemporalConvolution_single()
   local inputFrameSize = math.random(1,64)
   local outputFrameSize = math.random(1,64)
   local ki = math.random(1,15)
   local si = math.random(1,ki)
   local outi = math.random(1,15)
   local ini = (outi-1)*si+ki
   local scale = math.random()

   local input = torch.randn(ini,inputFrameSize):cuda()
   local gradOutput = torch.randn(outi,outputFrameSize):cuda()
   local sconv = nn.TemporalConvolution(inputFrameSize,outputFrameSize, ki, si):cuda()
   local groundForward = sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput, scale)
   cutorch.synchronize()
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias

   local gconv = cast(cudnn.TemporalConvolution(inputFrameSize,outputFrameSize, ki, si):cuda():fastest())
   gconv.weight:copy(sconv.weight:view(gconv.weight:size()))
   gconv.bias:copy(sconv.bias)
   gconv:forward(cast(input))

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   local cudaForward = gconv:forward(cast(input))
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(cast(input), cast(gradOutput), scale)
   cutorch.synchronize()
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias

   local ferror = cudaForward:float() - groundForward:float()
   local error = rescuda:float() - groundgrad:float()
   local werror = weightcuda:float() - groundweight:float()
   local berror = biascuda:float() - groundbias:float()
   mytester:assertlt(ferror:abs():max(), testparams.precision_forward, 'error on forward  ')
   mytester:assertlt(error:abs():max(), testparams.precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), testparams.precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), testparams.precision_backward, 'error on bias (backward) ')
end

function cudnntest.TemporalConvolution_reduceBatchSize()
   local inputFrameSize = math.random(1,64)
   local outputFrameSize = math.random(1,64)
   local ki = math.random(1,15)
   local si = math.random(1,ki)
   local outi = math.random(1,15)
   local ini = (outi-1)*si+ki
   local batchSize = 128
   local smallerBatchSize = batchSize/2

   local input = cast(torch.randn(batchSize,ini,inputFrameSize))
   local conv = cast(cudnn.TemporalConvolution(inputFrameSize,outputFrameSize,ki,si):cuda())
   local o1 = conv:updateOutput(input)
   mytester:asserteq(o1:size(1), batchSize, 'batch size didn\'t match')

   input = cast(torch.randn(smallerBatchSize,ini,inputFrameSize))
   local o2 = conv:updateOutput(input)
   mytester:asserteq(o2:size(1), smallerBatchSize, 'batch size didn\'t match')
   -- do this again to check it doesn't crash
   local o2 = conv:updateOutput(input)
   mytester:asserteq(o2:size(1), smallerBatchSize, 'batch size didn\'t match')
end


function cudnntest.VolumetricConvolution_forward_single()
   local from = math.random(1,16)
   local to = math.random(1,16)
   local ki = math.random(3,5)
   local kj = math.random(3,5)
   local kk = math.random(3,5)
   local si = math.random(1,ki-1)
   local sj = math.random(1,kj-1)
   local sk = math.random(1,kk-1)
   local outi = math.random(1,17)
   local outj = math.random(1,17)
   local outk = math.random(1,5)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk
   local input = torch.randn(from,ink,inj,ini):cuda()
   local sconv = nn.VolumetricConvolution(from,to,kk,ki,kj,sk,si,sj):float()
   local gconv = cast(cudnn.VolumetricConvolution(from,to,kk,ki,kj,sk,si,sj))
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   local function test(sconv, gconv)
     local groundtruth = sconv:forward(input:float())
     cutorch.synchronize()
     local rescuda = gconv:forward(cast(input))
     cutorch.synchronize()
     local error = rescuda:float() - groundtruth:float()
     mytester:assertlt(error:abs():max(), testparams.precision_forward,
                       'error on state (forward) ')

     -- IO
     local ferr,berr = jac.testIO(gconv, cast(input))
     mytester:assertlt(ferr, testparams.precision_io, torch.typename(gconv) .. ' - i/o forward err ')
     mytester:assertlt(berr, testparams.precision_io, torch.typename(gconv) .. ' - i/o backward err ')
   end

   test(sconv, gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn):cuda())
   mytester:asserteq(torch.typename(gconv), 'cudnn.VolumetricConvolution', 'conversion type check')
   test(sconv, gconv)
end

function cudnntest.VolumetricConvolution_backward_single()
   local from = math.random(1,16)
   local to = math.random(1,16)
   local ki = math.random(3,5)
   local kj = math.random(3,5)
   local kk = math.random(3,5)
   local si = math.random(1,ki-1)
   local sj = math.random(1,kj-1)
   local sk = math.random(1,kk-1)
   local outi = math.random(1,17)
   local outj = math.random(1,17)
   local outk = math.random(1,5)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk
   local input = torch.randn(from,ink,inj,ini):cuda()
   local gradOutput = torch.randn(to,outk,outj,outi):cuda()
   local sconv = nn.VolumetricConvolution(from,to,kk,ki,kj,sk,si,sj):float()
   sconv:forward(input:float())
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input:float(), gradOutput:float())
   cutorch.synchronize()
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias

   local gconv = cast(cudnn.VolumetricConvolution(from,to,kk,ki,kj,sk,si,sj))
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)

   local function test(sconv, gconv)
     gconv:forward(cast(input))
     cutorch.synchronize()

     -- serialize and deserialize
     torch.save('modelTemp.t7', gconv)
     gconv = torch.load('modelTemp.t7')

     gconv:forward(cast(input))
     gconv:zeroGradParameters()
     local rescuda = gconv:backward(cast(input), cast(gradOutput))
     cutorch.synchronize()

     mytester:asserteq(rescuda:dim(), 4, 'error in dimension')
     local weightcuda = gconv.gradWeight
     local biascuda = gconv.gradBias

     local error = rescuda:float() - groundgrad:float()
     local werror = weightcuda:float() - groundweight:float()
     local berror = biascuda:float() - groundbias:float()

     mytester:assertlt(error:abs():max(), testparams.precision_backward,
                       'error on state (backward) ')
     mytester:assertlt(werror:abs():max(), testparams.precision_backward,
                       'error on weight (backward) ')
     mytester:assertlt(berror:abs():max(), testparams.precision_backward,
                       'error on bias (backward) ')
  end

  test(sconv, gconv)
  local gconv = cast(cudnn.convert(sconv, cudnn):cuda())
  mytester:asserteq(torch.typename(gconv), 'cudnn.VolumetricConvolution', 'conversion type check')
  test(sconv, gconv)
end

function cudnntest.VolumetricMaxPooling_batch()
   if testparams.test_type ~= 'torch.CudaTensor' then return end
   local bs = math.random(1,4)
   local from = math.random(1,4)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local kk = math.random(2,4)
   local si = ki
   local sj = kj
   local sk = kk
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local outk = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk
   local input = torch.randn(bs,from,ink,inj,ini):cuda()
   local gradOutput = torch.randn(bs,from,outk,outj,outi):cuda()

   local sconv = nn.VolumetricMaxPooling(kk,ki,kj,sk,si,sj):float()
   local gconv = cast(cudnn.VolumetricMaxPooling(kk,ki,kj,sk,si,sj))
   local function test(sconv, gconv)
     local groundtruth = sconv:forward(input:float())
     local groundgrad = sconv:backward(input:float(), gradOutput:float())
     cutorch.synchronize()

     local rescuda = gconv:forward(cast(input))
     local resgrad = gconv:backward(cast(input), cast(gradOutput))
     cutorch.synchronize()

     mytester:asserteq(rescuda:dim(), 5, 'error in dimension')
     mytester:asserteq(resgrad:dim(), 5, 'error in dimension')
     local error = rescuda:float() - groundtruth:float()
     mytester:assertlt(error:abs():max(), testparams.precision_forward, 'error on state (forward) ')
     error = resgrad:float() - groundgrad:float()
     mytester:assertlt(error:abs():max(), testparams.precision_backward, 'error on state (backward) ')

     -- IO
     local ferr,berr = jac.testIO(gconv, cast(input))
     mytester:assertlt(ferr, testparams.precision_io, torch.typename(gconv) .. ' - i/o forward err ')
     mytester:assertlt(berr, testparams.precision_io, torch.typename(gconv) .. ' - i/o backward err ')
   end

   test(sconv, gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn):cuda())
   mytester:asserteq(torch.typename(gconv), 'cudnn.VolumetricMaxPooling', 'conversion type check')
   test(sconv, gconv)
end

function cudnntest.VolumetricMaxPooling_single()
   if not testparams.test_type == 'torch.CudaTensor' then return end
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local kk = math.random(2,4)
   local si = ki
   local sj = kj
   local sk = kk
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local outk = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk
   local input = torch.randn(from,ink,inj,ini):cuda()
   local gradOutput = torch.randn(from,outk,outj,outi):cuda()

   local sconv = nn.VolumetricMaxPooling(kk,ki,kj,sk,si,sj):float()
   local gconv = cast(cudnn.VolumetricMaxPooling(kk,ki,kj,sk,si,sj))

   local function test(sconv, gconv)
      local groundtruth = sconv:forward(input:float())
      local groundgrad = sconv:backward(input:float(), gradOutput:float())
      cutorch.synchronize()
      local _ = gconv:forward(cast(input))
      -- serialize and deserialize
      torch.save('modelTemp.t7', gconv)
      gconv = torch.load('modelTemp.t7')
      local rescuda = gconv:forward(cast(input))
      local resgrad = gconv:backward(cast(input), cast(gradOutput))
      cutorch.synchronize()
      mytester:asserteq(rescuda:dim(), 4, 'error in dimension')
      mytester:asserteq(resgrad:dim(), 4, 'error in dimension')
      local error = rescuda:float() - groundtruth:float()
      mytester:assertlt(error:abs():max(), testparams.precision_forward,
                        'error on state (forward) ')
      error = resgrad:float() - groundgrad:float()
      mytester:assertlt(error:abs():max(), testparams.precision_backward,
                        'error on state (backward) ')
   end

   test(sconv, gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn):cuda())
   mytester:asserteq(torch.typename(gconv), 'cudnn.VolumetricMaxPooling', 'conversion type check')
   test(sconv, gconv)
end

function cudnntest.SpatialMaxPooling_batch()
   if testparams.test_type ~= 'torch.CudaTensor' then return end
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(16,64)
   local outj = math.random(16,64)
   local padi = math.random(0,ki/2-1)
   local padj = math.random(0,kj/2-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1
   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,from,outj,outi):cuda()

   local sconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):cuda()
   if ceil_mode then sconv:ceil() end
   local groundtruth = sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local gconv = cast(cudnn.SpatialMaxPooling(ki,kj,si,sj,padi,padj))
   if ceil_mode then gconv:ceil() end
   local rescuda = gconv:forward(cast(input))
   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')
   local rescuda = gconv:forward(cast(input))
   local resgrad = gconv:backward(cast(input), cast(gradOutput))
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 4, 'error in dimension')
   mytester:asserteq(resgrad:dim(), 4, 'error in dimension')
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), testparams.precision_forward, 'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   mytester:assertlt(error:abs():max(), testparams.precision_backward, 'error on state (backward) ')

   -- IO
   local ferr,berr = jac.testIO(gconv, cast(input))
   mytester:assertlt(ferr, testparams.precision_io, torch.typename(gconv) .. ' - i/o forward err ')
   mytester:assertlt(berr, testparams.precision_io, torch.typename(gconv) .. ' - i/o backward err ')
end

function cudnntest.SpatialMaxPooling_single()
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(16,64)
   local outj = math.random(16,64)
   local padi = math.random(0,ki/2-1)
   local padj = math.random(0,kj/2-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1
   local input = torch.randn(from,inj,ini):cuda()
   local gradOutput = torch.randn(from,outj,outi):cuda()

   local sconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):cuda()
   if ceil_mode then sconv:ceil() end
   local gconv = cast(cudnn.SpatialMaxPooling(ki,kj,si,sj,padi,padj))
   if ceil_mode then gconv:ceil() end

   local function test(sconv, gconv)
      local groundtruth = sconv:forward(input)
      local groundgrad = sconv:backward(input, gradOutput)
      cutorch.synchronize()
      local _ = gconv:forward(cast(input))
      -- serialize and deserialize
      torch.save('modelTemp.t7', gconv)
      gconv = torch.load('modelTemp.t7')
      local rescuda = gconv:forward(cast(input))
      local resgrad = gconv:backward(cast(input), cast(gradOutput))
      cutorch.synchronize()
      mytester:asserteq(rescuda:dim(), 3, 'error in dimension')
      mytester:asserteq(resgrad:dim(), 3, 'error in dimension')
      local error = rescuda:float() - groundtruth:float()
      mytester:assertlt(error:abs():max(), testparams.precision_forward,
                        'error on state (forward) ')
      error = resgrad:float() - groundgrad:float()
      mytester:assertlt(error:abs():max(), testparams.precision_backward,
                        'error on state (backward) ')
   end

   test(sconv, gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn):cuda())
   mytester:asserteq(torch.typename(gconv), 'cudnn.SpatialMaxPooling', 'conversion type check')
   test(sconv, gconv)
end

function cudnntest.SpatialAveragePooling_batch()
   if testparams.test_type ~= 'torch.CudaTensor' then return end
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,from,outj,outi):cuda()

   local sconv = nn.SpatialAveragePooling(ki,kj,si,sj):cuda()
   local groundtruth = sconv:forward(input):clone()
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local gconv = cast(cudnn.SpatialAveragePooling(ki,kj,si,sj))
   local rescuda = gconv:forward(cast(input))
   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')
   local rescuda = gconv:forward(cast(input))
   local resgrad = gconv:backward(cast(input), gradOutput)
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 4, 'error in dimension')
   mytester:asserteq(resgrad:dim(), 4, 'error in dimension')
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), testparams.precision_forward, 'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   mytester:assertlt(error:abs():max(), testparams.precision_backward, 'error on state (backward) ')

   -- IO
   local ferr,berr = jac.testIO(gconv, cast(input))
   mytester:assertlt(ferr, testparams.precision_io, torch.typename(gconv) .. ' - i/o forward err ')
   mytester:assertlt(berr, testparams.precision_io, torch.typename(gconv) .. ' - i/o backward err ')
end

function cudnntest.SpatialAveragePooling_single()
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local input = torch.randn(from,inj,ini):cuda()
   local gradOutput = torch.randn(from,outj,outi):cuda()

   local sconv = nn.SpatialAveragePooling(ki,kj,si,sj):cuda()
   local gconv = cast(cudnn.SpatialAveragePooling(ki,kj,si,sj))

   mytester:assert(cudnn.C.CUDNN_POOLING_AVERAGE ~= nil, 'back-compat broken')

   local function test(sconv, gconv)
      local groundtruth = sconv:forward(input):clone()
      local groundgrad = sconv:backward(input, gradOutput)
      cutorch.synchronize()
      local _ = gconv:forward(cast(input))
      -- serialize and deserialize
      torch.save('modelTemp.t7', gconv)
      gconv = torch.load('modelTemp.t7')
      local rescuda = gconv:forward(cast(input))
      local resgrad = gconv:backward(cast(input), cast(gradOutput))
      cutorch.synchronize()
      mytester:asserteq(rescuda:dim(), 3, 'error in dimension')
      mytester:asserteq(resgrad:dim(), 3, 'error in dimension')
      local error = rescuda:float() - groundtruth:float()
      mytester:assertlt(error:abs():max(), testparams.precision_forward,
                        'error on state (forward) ')
      error = resgrad:float() - groundgrad:float()
      mytester:assertlt(error:abs():max(), testparams.precision_backward,
                        'error on state (backward) ')
   end

   test(sconv, gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn):cuda())
   mytester:asserteq(torch.typename(gconv), 'cudnn.SpatialAveragePooling', 'conversion type check')
   test(sconv, gconv)
end

local function nonlinSingle(nonlin)
   local from = math.random(1,32)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = outi
   local inj = outj
   local input = torch.randn(from,inj,ini):cuda()
   local gradOutput = torch.randn(from,outj,outi):cuda()

   local sconv = nn[nonlin]():cuda()
   local gconv = cast(cudnn[nonlin](inplace))
   local function test(sconv, gconv)
     local groundtruth = sconv:forward(input)
     local groundgrad = sconv:backward(input, gradOutput)
     cutorch.synchronize()
     -- 50% prob to choose inplace or out-of-place
     local inplace = false
     if math.random(0,1) == 1 then
        inplace = true
     end
     local input__ = cast(input:clone())
     local _ = gconv:forward(input__)

     -- serialize and deserialize
     torch.save('modelTemp.t7', gconv)
     gconv = torch.load('modelTemp.t7')

     local input__ = cast(input:clone())
     local gradOutput__ = cast(gradOutput:clone())
     local rescuda = gconv:forward(input__)
     local resgrad = gconv:backward(input__, gradOutput__)
     cutorch.synchronize()
     mytester:asserteq(rescuda:dim(), 3, 'error in dimension')
     mytester:asserteq(resgrad:dim(), 3, 'error in dimension')
     local error = rescuda:float() - groundtruth:float()
     mytester:assertlt(error:abs():max(), testparams.precision_forward,
                       'error on state (forward) ')
     error = resgrad:float() - groundgrad:float()
     mytester:assertlt(error:abs():max(), testparams.precision_backward,
                       'error on state (backward) ')
     -- IO
     local ferr,berr = jac.testIO(gconv, cast(input))
     mytester:assertlt(ferr, testparams.precision_io, torch.typename(gconv) .. ' - i/o forward err ')
     mytester:assertlt(berr, testparams.precision_io, torch.typename(gconv) .. ' - i/o backward err ')
   end

   test(sconv, gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv), 'cudnn.'..nonlin, 'conversion type check')
   test(sconv, gconv)
end

local function nonlinBatch(nonlin)
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = outi
   local inj = outj
   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,from,outj,outi):cuda()

   local sconv = nn[nonlin]():cuda()
   local gconv = cast(cudnn[nonlin](inplace))
   local function test(sconv, gconv)
      local groundtruth = sconv:forward(input)
      local groundgrad = sconv:backward(input, gradOutput)
      cutorch.synchronize()
      -- 50% prob to choose inplace or out-of-place
      local inplace = false
      if math.random(0,1) == 1 then
         inplace = true
      end
      local input__ = cast(input:clone())
      local rescuda = gconv:forward(input__)

      -- serialize and deserialize
      torch.save('modelTemp.t7', gconv)
      gconv = torch.load('modelTemp.t7')

      local input__ = cast(input:clone())
      local gradOutput__ = cast(gradOutput:clone())
      local rescuda = gconv:forward(input__)
      local resgrad = gconv:backward(input__, gradOutput__)
      cutorch.synchronize()
      mytester:asserteq(rescuda:dim(), 4, 'error in dimension')
      mytester:asserteq(resgrad:dim(), 4, 'error in dimension')
      local error = rescuda:float() - groundtruth:float()
      mytester:assertlt(error:abs():max(), testparams.precision_forward,
                        'error on state (forward) ')
      error = resgrad:float() - groundgrad:float()
      mytester:assertlt(error:abs():max(), testparams.precision_backward,
                        'error on state (backward) ')
     -- IO
     local ferr,berr = jac.testIO(gconv, cast(input))
     mytester:assertlt(ferr, testparams.precision_io, torch.typename(gconv) .. ' - i/o forward err ')
     mytester:assertlt(berr, testparams.precision_io, torch.typename(gconv) .. ' - i/o backward err ')
   end

   test(sconv, gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv), 'cudnn.'..nonlin, 'conversion type check')
   test(sconv, gconv)
end

function cudnntest.ReLU_single()
   nonlinSingle('ReLU')
end

function cudnntest.ReLU_batch()
   nonlinBatch('ReLU')
end

function cudnntest.ClippedReLU_single()
    local input = torch.randn(1, 32):cuda()
    local ceiling = 0.1
    local module = cudnn.ClippedReLU(ceiling):cuda()
    local output = module:forward(input)
    local expectedOutput = input:clone()
    expectedOutput[expectedOutput:ge(ceiling)] = ceiling
    expectedOutput[expectedOutput:le(0)] = 0
    mytester:assertTensorEq(output, expectedOutput)
end

function cudnntest.Tanh_single()
   nonlinSingle('Tanh')
end

function cudnntest.Tanh_batch()
   nonlinBatch('Tanh')
end

function cudnntest.Sigmoid_single()
   nonlinSingle('Sigmoid')
end

function cudnntest.Sigmoid_batch()
   nonlinBatch('Sigmoid')
end

function cudnntest.SpatialCrossMapLRN_batch()
   local bs = math.random(4,10)
   local inputSize = math.random(6,9)
   local size = math.random(1,3)*2+1
   local nbfeatures = math.random(3,8)
   local alpha = math.random(1,100)/100
   local beta  = math.random(0,100)/100
   local k = math.random(1,3)

   local tm = {}
   local title = string.format('SpatialCrossMapLRN.forward')
   times[title] = tm

   local input = torch.rand(bs, nbfeatures, inputSize, inputSize):cuda()
   local gradOutput = torch.rand(input:size()):cuda()
   local sconv = nn.SpatialCrossMapLRN(size, alpha, beta, k):cuda()
   local gconv = cast(cudnn.SpatialCrossMapLRN(size, alpha, beta, k))

   local function test(sconv, gconv)
     local groundtruth = sconv:forward(input):clone()
     local groundgrad = sconv:backward(input, gradOutput)
     cutorch.synchronize()
     gconv:forward(cast(input))
     -- serialize and deserialize
     torch.save('modelTemp.t7', gconv)
     gconv = torch.load('modelTemp.t7')
     local rescuda = gconv:forward(cast(input))
     local resgrad = gconv:backward(cast(input), cast(gradOutput))
     cutorch.synchronize()
     mytester:asserteq(rescuda:dim(), 4, 'error in dimension')
     mytester:asserteq(resgrad:dim(), 4, 'error in dimension')
     local error = rescuda:float() - groundtruth:float()
     mytester:assertlt(error:abs():max(), testparams.precision_forward,
                       'error on state (forward) ')
     error = resgrad:float() - groundgrad:float()
     mytester:assertlt(error:abs():max(), testparams.precision_backward,
                       'error on state (backward) ')
   end

   test(sconv, gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv), 'cudnn.SpatialCrossMapLRN', 'conversion type check')
   test(sconv, gconv)
end


function cudnntest.SoftMax_single()
   local sz = math.random(1,64)
   local input = torch.randn(sz):cuda()
   local gradOutput = torch.randn(sz):cuda()

   local sconv = nn.SoftMax():cuda()
   local groundtruth = sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local gconv = cast(cudnn.SoftMax())
   local _ = gconv:forward(cast(input))

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   local rescuda = gconv:forward(cast(input))
   local resgrad = gconv:backward(cast(input), cast(gradOutput))
   cutorch.synchronize()
   local error = rescuda:float() - groundtruth:float()
   local errmax = error:abs():max()
   if (errmax ~= errmax) then
      local state = {}
      state.input = input
      state.gradOutput = gradOutput
      state.rescuda = rescuda
      state.resgrad = resgrad
      state.groundtruth = groundtruth
      state.groundgrad = groundgrad
      print(#input)
      torch.save('badSoftMax.t7', state)
      print(#input)
   end
   mytester:assertlt(errmax, testparams.precision_forward,
                     'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   errmax = error:abs():max()
   if (errmax ~= errmax) then
      local state = {}
      state.input = input
      state.gradOutput = gradOutput
      state.rescuda = rescuda
      state.resgrad = resgrad
      state.groundtruth = groundtruth
      state.groundgrad = groundgrad
      print(#input)
      torch.save('badSoftMax.t7', state)
      print(#input)
   end
   mytester:assertlt(errmax, testparams.precision_backward,
                     'error on state (backward) ')
end

function cudnntest.SoftMax_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = outi
   local inj = outj
   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,from,outj,outi):cuda()

   local sconv = nn.SoftMax():cuda()
   local groundtruth = sconv:forward(input:view(bs,-1))
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local gconv = cast(cudnn.SoftMax())
   local rescuda = gconv:forward(cast(input))

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   local rescuda = gconv:forward(cast(input))
   local resgrad = gconv:backward(cast(input), cast(gradOutput))
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 4, 'error in dimension')
   mytester:asserteq(resgrad:dim(), 4, 'error in dimension')

   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(),
                     testparams.precision_forward, 'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   mytester:assertlt(error:abs():max(),
                     testparams.precision_backward, 'error on state (backward) ')

   -- IO
   local ferr,berr = jac.testIO(gconv, cast(input))
   mytester:assertlt(ferr, testparams.precision_io, torch.typename(gconv) .. ' - i/o forward err ')
   mytester:assertlt(berr, testparams.precision_io, torch.typename(gconv) .. ' - i/o backward err ')
end


function cudnntest.LogSoftMax_single()
   local sz = math.random(1,64)
   local input = torch.randn(sz):cuda()
   local gradOutput = torch.randn(sz):cuda()

   local sconv = nn.LogSoftMax():cuda()
   local gconv = cast(cudnn.LogSoftMax())

   local function test(sconv, gconv)
      local groundtruth = sconv:forward(input)
      local groundgrad = sconv:backward(input, gradOutput)
      cutorch.synchronize()
      local _ = gconv:forward(cast(input))

      -- serialize and deserialize
      torch.save('modelTemp.t7', gconv)
      gconv = torch.load('modelTemp.t7')

      local rescuda = gconv:forward(cast(input))
      local resgrad = gconv:backward(cast(input), cast(gradOutput))
      cutorch.synchronize()
      local error = rescuda:float() - groundtruth:float()
      local errmax = error:abs():max()
      mytester:assertlt(errmax, testparams.precision_forward,
                        'error on state (forward) ')
      error = resgrad:float() - groundgrad:float()
      errmax = error:abs():max()
      mytester:assertlt(errmax, testparams.precision_backward,
                        'error on state (backward) ')
      -- IO
      local ferr,berr = jac.testIO(gconv, cast(input))
      mytester:assertlt(ferr, testparams.precision_io, torch.typename(gconv) .. ' - i/o forward err ')
      mytester:assertlt(berr, testparams.precision_io, torch.typename(gconv) .. ' - i/o backward err ')
   end

   test(sconv, gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv), 'cudnn.LogSoftMax', 'conversion type check')
   test(sconv, gconv)
end

function cudnntest.LogSoftMax_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local input = torch.randn(bs,from):cuda()
   local gradOutput = torch.randn(bs,from):cuda()

   local sconv = nn.LogSoftMax():cuda()
   local gconv = cast(cudnn.LogSoftMax())
   local function test(sconv, gconv)
      local groundtruth = sconv:forward(input)
      local groundgrad = sconv:backward(input, gradOutput)
      cutorch.synchronize()
      local rescuda = gconv:forward(cast(input))

      -- serialize and deserialize
      torch.save('modelTemp.t7', gconv)
      gconv = torch.load('modelTemp.t7')

      local rescuda = gconv:forward(cast(input))
      local resgrad = gconv:backward(cast(input), cast(gradOutput))
      cutorch.synchronize()
      mytester:asserteq(rescuda:dim(), 2, 'error in dimension')
      mytester:asserteq(resgrad:dim(), 2, 'error in dimension')

      local error = rescuda:float() - groundtruth:float()
      mytester:assertlt(error:abs():max(),
                        testparams.precision_forward, 'error on state (forward) ')
      error = resgrad:float() - groundgrad:float()
      mytester:assertlt(error:abs():max(),
                        testparams.precision_backward, 'error on state (backward) ')

      -- IO
      local ferr,berr = jac.testIO(gconv, cast(input))
      mytester:assertlt(ferr, testparams.precision_io, torch.typename(gconv) .. ' - i/o forward err ')
      mytester:assertlt(berr, testparams.precision_io, torch.typename(gconv) .. ' - i/o backward err ')
   end

   test(sconv, gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv), 'cudnn.LogSoftMax', 'conversion type check')
   test(sconv, gconv)
end

function cudnntest.SpatialLogSoftMax()
    -- batch
    local numLabels = math.random(5,10)
    local h = math.random(5,10)
    local w = math.random(5,10)
    local bsz = math.random(3, 7)
    local input = torch.zeros(bsz, numLabels, h, w):normal():cuda()
    local target = torch.zeros(bsz, numLabels, h, w):normal():cuda()

    local cri = cast(cudnn.SpatialLogSoftMax())
    local gcri = nn.LogSoftMax():cuda()

    local op = cri:forward(cast(input), cast(target))
    local gi = cri:backward(cast(input), cast(target))

    local gop = op:clone():zero()
    local ggi = gi:clone():zero()

    for i=1,h do
        for j=1,w do
            local i1 = input[{{}, {}, {i}, {j}}]:contiguous():squeeze()
            local t1 = target[{{}, {}, {i}, {j}}]:contiguous():squeeze()
            local gop1 = gcri:forward(i1, t1)
            local ggi1 = gcri:backward(i1, t1)
            gop[{{}, {}, {i}, {j}}]:copy(gop1)
            ggi[{{}, {}, {i}, {j}}]:copy(ggi1)
        end
    end
    local err = (gi - ggi):abs():max()
    mytester:assertlt(err, testparams.precision_backward, 'error in difference between central difference and :backward')
    local err = (op - gop):abs():max()
    mytester:assertlt(err, testparams.precision_backward, 'error in difference between central difference and :backward')
end

local function testBatchNormalization(moduleName, inputSize)
   local input = torch.randn(table.unpack(inputSize)):cuda()
   local gradOutput = torch.randn(table.unpack(inputSize)):cuda()
   local cbn = cast(cudnn[moduleName](inputSize[2], 1e-3))
   local gbn = nn[moduleName](inputSize[2], 1e-3):cuda()
   cbn.weight:copy(gbn.weight)
   cbn.bias:copy(gbn.bias)

   local function testFWDBWD(cbn, gbn)
      cbn:training()
      gbn:training()
      mytester:asserteq(cbn.running_mean:mean(), 0, 'error on BN running_mean init')
      mytester:asserteq(cbn.running_var:mean(), 1, 'error on BN running_var init')
      local rescuda = cbn:forward(cast(input))
      local groundtruth = gbn:forward(input)
      local resgrad = cbn:backward(cast(input), cast(gradOutput))
      local groundgrad = gbn:backward(input, gradOutput)

      local error = rescuda:float() - groundtruth:float()
      mytester:assertlt(error:abs():max(),
         testparams.precision_forward, 'error in batch normalization (forward) ')
      error = resgrad:float() - groundgrad:float()
      mytester:assertlt(error:abs():max(),
         testparams.precision_backward, 'error in batch normalization (backward) ')
      error = cbn.running_mean:float() - gbn.running_mean:float()
      mytester:assertlt(error:abs():max(),
         testparams.precision_forward, 'error in batch normalization (running_mean) ')
      error = cbn.running_var:float() - gbn.running_var:float()
      mytester:assertlt(error:abs():max(),
         testparams.precision_forward, 'error in batch normalization (running_var) ')
   end

   local function testFWD(cbn, gbn)
      cbn:evaluate()
      gbn:evaluate()
      local rescuda = cbn:forward(cast(input))
      local groundtruth = gbn:forward(input)

      local error = rescuda:float() - groundtruth:float()
      mytester:assertlt(error:abs():max(),
         testparams.precision_forward, 'error in batch normalization (forward) ')
   end

   testFWDBWD(cbn, gbn)
   testFWD(cbn, gbn)
   if testparams.test_type == 'torch.CudaTensor' then
      local cudnn2nn = cast(cudnn.convert(cbn:clone(), nn))
      mytester:asserteq(torch.type(cudnn2nn), 'nn.'..moduleName, 'cudnn to nn')
      testFWD(cudnn2nn, gbn)
      local nn2cudnn = cast(cudnn.convert(gbn:clone(), cudnn))
      mytester:asserteq(torch.type(nn2cudnn), 'cudnn.'..moduleName, 'cudnn to nn')
      testFWD(nn2cudnn, gbn)
   end
end

function cudnntest.BatchNormalization()
   local size = {
      math.random(2, 32),
      math.random(16, 256),
   }
   testBatchNormalization('BatchNormalization', size)
end

function cudnntest.SpatialBatchNormalization()
   local size = {
      math.random(1, 32),
      math.random(1, 32),
      math.random(5, 10),
      math.random(5, 10),
   }
   testBatchNormalization('SpatialBatchNormalization', size)
end

function cudnntest.VolumetricBatchNormalization()
   local size = {
      math.random(1, 32),
      math.random(1, 32),
      math.random(2, 6),
      math.random(2, 6),
      math.random(2, 6),
   }
   testBatchNormalization('VolumetricBatchNormalization', size)
end

function cudnntest.SpatialCrossEntropyCriterion()
    if testparams.test_type ~= 'torch.CudaTensor' then return end
    -- batch
    local numLabels = math.random(5,10)
    local h = math.random(5,10)
    local w = math.random(5,10)
    local bsz = math.random(3, 7)
    local input = torch.zeros(bsz, numLabels, h, w):normal():cuda()
    local target = torch.Tensor(bsz, h, w):random(1, numLabels):cuda()

    local cri = cast(cudnn.SpatialCrossEntropyCriterion())

    local gcri = nn.CrossEntropyCriterion():cuda()

    local op = cri:forward(cast(input), cast(target))
    local gi = cri:backward(cast(input), cast(target))

    local ggi = gi:clone():zero()

    for i=1,h do
        for j=1,w do
            local i1 = input[{{}, {}, {i}, {j}}]:contiguous():squeeze()
            local t1 = target[{{}, {i}, {j}}]:contiguous():squeeze()
            local gop1 = gcri:forward(i1, t1)
            local ggi1 = gcri:backward(i1, t1)
            ggi[{{}, {}, {i}, {j}}]:copy(ggi1)
        end
    end

    -- nn.CrossEntropy in contrast to cudnn.SpatialCrossEntropyCriterion cannot
    -- average over the last spatial dimensions because it is run in a loop
    ggi:div(h * w)

    local err = (gi - ggi):abs():max()
    mytester:assertlt(err, testparams.precision_backward, 'error in difference between central difference and :backward')

end

function cudnntest.functional_bias2D()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local scale = torch.uniform()
   local input = torch.zeros(bs,from,inj,ini):cuda()
   local mod = cudnn.SpatialConvolution(from,to,ki,kj,si,sj):cuda()
   mod.weight:zero()
   local groundtruth = mod:forward(input)
   local result = groundtruth:clone():zero()
   cudnn.functional.bias2D_updateOutput(cudnn.getHandle(), mod.bias, result)
   local error = result:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(),
                     testparams.precision_forward, 'error on forward ')

   mod:zeroGradParameters()
   local gradOutput = groundtruth:clone():normal()
   mod:backward(input, gradOutput, scale)
   local groundtruth = mod.gradBias
   local result = groundtruth:clone():zero()
   cudnn.functional.bias2D_accGradParameters(cudnn.getHandle(), gradOutput, result, scale)
   error = result:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(),
                     testparams.precision_backward, 'error on accGradParameters ')
end
function cudnntest.functional_maxpooling2d()
    local a=cudnn.SpatialMaxPooling(2,2,2,2):cuda()
    local input = torch.randn(10,3,10,10):cuda()
    a:forward(input);
    local output = a.output:clone():normal()
    local gradOutput = a.output:clone():normal()
    local gradInput = a:backward(input, gradOutput):clone():normal()
    cudnn.functional.MaxPooling2D_updateOutput(cudnn.getHandle(), input,
                                               output, a.kH, a.kW,
                                               a.dH, a.dW, a.padH, a.padW)
    mytester:assertlt((output - a.output):abs():max(),
                     testparams.precision_forward, 'error on forward ')

    cudnn.functional.MaxPooling2D_updateGradInput(cudnn.getHandle(), input,
                                                   output, gradOutput, gradInput,
                                                   a.kH, a.kW, a.dH, a.dW,
                                                   a.padH, a.padW)
    mytester:assertlt((gradInput - a.gradInput):abs():max(),
                     testparams.precision_forward, 'error on updateGradInput ')
end

torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())
mytester = torch.Tester()
mytester:add(cudnntest)

if torch.random(1,2) == 1 then
   cudnn.benchmark = true -- run manual auto-tuner
   cudnn.verbose = true
end


for i=1,cutorch.getDeviceCount() do
   local prop = cutorch.getDeviceProperties(i)

   print('Running test on device: #' .. i .. ' : ' .. prop.name)

   cutorch.setDevice(i)
   print'Testing torch.CudaTensor'
   testparams = testparams_float
   mytester:run()

   print'Testing torch.CudaHalfTensor'
   testparams = testparams_half
   mytester:run()

#   print'Testing torch.CudaDoubleTensor'
#   testparams = testparams_double
#   mytester:run()
end

os.execute('rm -f modelTemp.t7')
