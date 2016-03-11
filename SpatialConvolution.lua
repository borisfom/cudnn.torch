local SpatialConvolution, parent =
    torch.class('cudnn.SpatialConvolution', 'nn.SpatialConvolution')

local ffi = require 'ffi'
local errcheck = cudnn.errcheck
local impl = cudnn.ConvolutionImpl()

function SpatialConvolution:__init(nInputPlane, nOutputPlane,
                            kW, kH, dW, dH, padW, padH, groups)
    impl.init(self, parent, nInputPlane, nOutputPlane, kW, kH, dW, dH)
end

-- if you change the configuration of the module manually, call this
function SpatialConvolution:resetWeightDescriptors()
    impl.resetWeightDescriptors(self)
end

function SpatialConvolution:fastest(mode)
   return impl.fastest(self)
end

function SpatialConvolution:setMode(fmode, bdmode, bwmode)
    return impl.setMode(self, fmode, bdmode, bwmode)
end

function SpatialConvolution:resetMode()
    return impl.resetMode(self)
end

function SpatialConvolution:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function SpatialConvolution:createIODescriptors(input)
    return impl.createIODescriptors(self, input)
end

function SpatialConvolution:updateOutput(input)
   if not self.weightDesc then self:resetWeightDescriptors() end
   input = impl.makeContiguous(self, input)
   return impl.updateOutput(self, input)
end

function SpatialConvolution:updateGradInput(input, gradOutput)
    if not self.gradInput then return end
    input, gradOutput = impl.makeContiguous(self, input, gradOutput)
    return impl.updateGradInput(self, input, gradOutput)
end

function SpatialConvolution:accGradParameters(input, gradOutput, scale)
    input, gradOutput = impl.makeContiguous(self, input, gradOutput)
    impl.accGradParameters(self, input, gradOutput, scale)
end

function SpatialConvolution:clearDesc()
   impl.clearDesc(self)
end

function SpatialConvolution:write(f)
   impl.write(self, f)
end

function SpatialConvolution:clearState()
   impl.clearState(self)
end
