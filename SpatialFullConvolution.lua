local SpatialFullConvolution, parent =
    torch.class('cudnn.SpatialFullConvolution', 'nn.SpatialFullConvolution')
local ffi = require 'ffi'
local impl = require 'ConvolutionImpl'

local errcheck = cudnn.errcheck

function SpatialFullConvolution:__init(nInputPlane, nOutputPlane,
                            kW, kH, dW, dH, padW, padH, groups)
    impl.init(self, parent, nInputPlane, nOutputPlane, kW, kH, dW, dH)
end

-- if you change the configuration of the module manually, call this
function SpatialFullConvolution:resetWeightDescriptors()
    assert(self.bias, 'SpatialFullConvolution assumes bias!=nil')
    impl.resetWeightDescriptors(self)
end

function SpatialFullConvolution:fastest(mode)
   return impl.fastest(self)
end

function SpatialFullConvolution:setMode(fmode, bdmode, bwmode)
    return impl.setMode(self, fmode, bdmode, bwmode)
end

function SpatialFullConvolution:resetMode()
    return impl.resetMode(self)
end

function SpatialFullConvolution:createIODescriptors(input)
    return impl.createIODescriptors(self, input)
end

function SpatialFullConvolution:updateOutput(input)
   return impl.updateOutput(self, input)
end

function SpatialFullConvolution:updateGradInput(input, gradOutput)
    return impl.updateGradInput(self, input, gradOutput)
end

function SpatialFullConvolution:accGradParameters(input, gradOutput, scale)
    impl.accGradParameters(self, input, gradOutput, scale)
end

function SpatialFullConvolution:clearDesc()
   impl.clearDesc(self)
end

function SpatialFullConvolution:write(f)
   impl.write(self, f)
end

function SpatialFullConvolution:clearState()
   impl.clearState(self)
end
