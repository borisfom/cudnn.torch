local VolumetricConvolution, parent
   = torch.class('cudnn.VolumetricConvolution', 'nn.VolumetricConvolution')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck
local impl = cudnn.ConvolutionImpl()

function VolumetricConvolution:__init(nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH)
    impl.init3D(self, parent, nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH)
end

-- if you change the configuration of the module manually, call this
function VolumetricConvolution:resetWeightDescriptors()
   impl.resetWeightDescriptors(self)
end

function VolumetricConvolution:fastest(mode)
   return impl.fastest(self, mode);
end

function VolumetricConvolution:setMode(fmode, bdmode, bwmode)
   return impl.setmode(self, fmode, bdmode, bwmode)
end

function VolumetricConvolution:resetMode()
   return impl.resetMode()
end

function VolumetricConvolution:createIODescriptors(input)
   if input:dim() == 4 then
      input = input:view(1, input:size(1), input:size(2),
                         input:size(3), input:size(4))
   end
   assert(input:dim() == 5 and input:isContiguous());
   return impl.createIODescriptors(self, input);
end

function VolumetricConvolution:updateOutput(input)
   input = impl.makeContiguous(self, input)
   return impl.updateOutput(self, input)
end

function VolumetricConvolution:updateGradInput(input, gradOutput)
   if not self.gradInput then return end
   input, gradOutput = impl.makeContiguous(self, input, gradOutput)
   return impl.updateGradInput(self, input, gradOutput)
end

function VolumetricConvolution:accGradParameters(input, gradOutput, scale)
   input, gradOutput = impl.makeContiguous(self, input, gradOutput)
   return impl.accGradParameters(self,input, gradOutput, scale)
end

function VolumetricConvolution:clearDesc()
   return impl.clearDesc(self)
end

function VolumetricConvolution:write(f)
   return impl.write(self, f)
end

function VolumetricConvolution:clearState()
   return impl.clearState(self)
end
