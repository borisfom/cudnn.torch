local ConvolutionImpl = torch.class('cudnn.ConvolutionImpl')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

local autotunerCache = {}
autotunerCache[1] = {} -- forward
autotunerCache[2] = {} -- backwardFilter
autotunerCache[3] = {} -- backwardData

function ConvolutionImpl:__init()
end

function ConvolutionImpl:init(parent, nInputPlane, nOutputPlane,
                            kW, kH, dW, dH, padW, padH, groups)
    local delayedReset = self.reset
    self.reset = function() end
    parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    self.reset = delayedReset
    self.padW = padW or 0
    self.padH = padH or 0
    self.groups = groups or 1
    assert(nInputPlane % self.groups == 0,
           'nInputPlane should be divisible by nGroups')
    assert(nOutputPlane % self.groups == 0,
           'nOutputPlane should be divisible by nGroups')
    self.weight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
    self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
    self:reset()
    -- should nil for serialization, the reset will still work
    self.reset = nil
    self.nDim = 4
    self.iSize = torch.LongStorage(4):fill(0)
    return self
end


function ConvolutionImpl:init3D(parent, nInputPlane, nOutputPlane,
                              kT, kW, kH, dT, dW, dH, padT, padW, padH, groups)
    local delayedReset = self.reset
    self.reset = function() end
    parent.__init(self, nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH)
    self.reset = delayedReset
    self.padW = padW or 0
    self.padH = padH or 0
    self.padT = padT or 0

    self.groups = groups or 1
    self.input_offset = 0
    self.output_offset = 0
    self.weight_offset = 0

    assert(nInputPlane % self.groups == 0,
           'nInputPlane should be divisible by nGroups')
    assert(nOutputPlane % self.groups == 0,
           'nOutputPlane should be divisible by nGroups')
    self.weight = torch.Tensor(nOutputPlane, nInputPlane, kT, kH, kW)
    self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kT, kH, kW)
    self:reset()
    -- should nil for serialization, the reset will still work
    self.reset = nil
    self.iSize = torch.LongStorage(5):fill(0)
    self.nDim = 5
    return self
end

-- if you change the configuration of the module manually, call this
function ConvolutionImpl:resetWeightDescriptors()
    assert(torch.typename(self.weight) == 'torch.CudaTensor',
           'Only Cuda supported duh!')
    assert(torch.typename(self.bias) == 'torch.CudaTensor' or not self.bias,
           'Only Cuda supported duh!')
    -- create filterDescriptor for weight
    self.weightDesc = ffi.new('struct cudnnFilterStruct*[1]')
    errcheck('cudnnCreateFilterDescriptor', self.weightDesc)
    local desc = nil
    local mydim =  self.nDim or 4

    if mydim == 4 then
       -- for compatibility
       self.groups = self.groups or 1
       desc = torch.IntTensor({self.nOutputPlane/self.groups,
                               self.nInputPlane/self.groups,
                               self.kH, self.kW})
    elseif mydim == 5 then
       -- no groups here
       desc = torch.IntTensor({self.nOutputPlane, self.nInputPlane,
                               self.kT, self.kH, self.kW})
    end
    errcheck('cudnnSetFilterNdDescriptor', self.weightDesc[0],
             'CUDNN_DATA_FLOAT', 'CUDNN_TENSOR_NCHW', mydim,
             desc:data());
    local function destroyWDesc(d)
        errcheck('cudnnDestroyFilterDescriptor', d[0]);
    end
    ffi.gc(self.weightDesc, destroyWDesc)

    -- create descriptor for bias
    if self.bias then
        self.biasDesc = cudnn.toDescriptor(self.bias:view(1, self.nOutputPlane,1,1))
    end
    return self
end

function ConvolutionImpl:fastest(mode)
   self.fastest_mode = mode or true
   return self
end

function ConvolutionImpl:setMode(fmode, bdmode, bwmode)
    if fmode ~= nil then
        self.fmode = fmode
    end
    if bdmode ~= nil then
        self.bdmode = bdmode
    end
    if bwmode ~= nil then
        self.bwmode = bwmode
    end
    return self
end

function ConvolutionImpl:resetMode()
    self.fmode = nil
    self.bdmode = nil
    self.bwmode = nil
    return self
end

function ConvolutionImpl:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

local function arraysEqual(a1, a2)
   if not a1 or #a1 then return false end
   for i=1,#a1 do
      if a1[1] ~= a2[1] then
         return false
      end
   end
   return true
end

-- returns true if size changed
function ConvolutionImpl:createIODescriptors(input)
    local batch = true
    local mydim = self.nDim or 4

    if mydim == 4 then
       if input:dim() == 3 then
          input = input:view(1, input:size(1), input:size(2), input:size(3))
          batch = false
       end
    end
    assert(input:dim() == mydim and input:isContiguous());

    if self.iDesc and self.oDesc and arraysEqual(iSize, input:size()) then
       return false
    end

    self.iSize = input:size()
    -- resize gradInput
    if self.gradInput then self.gradInput:resizeAs(input) end
    assert(self.nInputPlane == input:size(2), 'input has to contain: '
              .. self.nInputPlane
              .. ' feature maps, but received input of size: '
              .. input:size(1) .. ' x ' .. input:size(2) ..
              ' x ' .. input:size(3) .. ' x ' .. input:size(4))

    local input_slice = nil
    -- create input descriptor
    if mydim == 5 then
       input_slice = {{},{1,self.nInputPlane/self.groups},{},{}, {}}
       self.iDesc = cudnn.toDescriptor(input[input_slice])
    else
         input_slice = {{},{1,self.nInputPlane/self.groups},{},{}}
         self.iDesc = cudnn.toDescriptor(input)
    end
    -- create conv descriptor
    self.convDesc = ffi.new('struct cudnnConvolutionStruct*[1]')
    errcheck('cudnnCreateConvolutionDescriptor', self.convDesc)

    local pad = torch.IntTensor({self.padH, self.padW})
    local stride = torch.IntTensor({self.dH, self.dW})
    local upscale = torch.IntTensor({1,1})
    if  mydim == 5 then
          pad = torch.IntTensor({self.padT, self.padH, self.padW})
          stride = torch.IntTensor({self.dT, self.dH, self.dW})
          upscale = torch.IntTensor({1,1,1})
    else
       pad = torch.IntTensor({self.padH, self.padW})
       stride = torch.IntTensor({self.dH, self.dW})
       upscale = torch.IntTensor({1,1})
    end

    errcheck('cudnnSetConvolutionNdDescriptor', self.convDesc[0],
             mydim-2, pad:data(),
             stride:data(), upscale:data(), 'CUDNN_CROSS_CORRELATION',
             'CUDNN_DATA_FLOAT');
    local function destroyConvDesc(d)
       errcheck('cudnnDestroyConvolutionDescriptor', d[0]);
    end
    ffi.gc(self.convDesc, destroyConvDesc)

    -- get output shape, resize output
    local oSize = torch.IntTensor(mydim)
    local oSizeD = oSize:data()
    errcheck('cudnnGetConvolutionNdForwardOutputDim',
             self.convDesc[0], self.iDesc[0],
             self.weightDesc[0], mydim, oSizeD)
    oSize[2] = oSize[2] * self.groups
    self.output:resize(oSize:long():storage())
    local output_slice;
    output_slice = {{},{1,self.nOutputPlane/self.groups},{},{}}
    -- create descriptor for output
    if mydim ==5 then
       self.oDesc = cudnn.toDescriptor(self.output)
       self.oDescForBias = cudnn.toDescriptor(
          self.output:view(self.output:size(1),
                           self.output:size(2),
                           self.output:size(3)*self.output:size(4),
                           self.output:size(5)))
    else
       self.oDescForBias = cudnn.toDescriptor(self.output)
       self.oDesc = cudnn.toDescriptor(self.output[output_slice])
    end

    -----------------------------------------------------------------------
    local function shape(x)
       local sz = x:size()
       local str = ''
       for i=1,sz:size() do
          str = str .. sz[i] .. 'x'
       end
       if #str > 0 then
          str = str:sub(1, #str-1)
       end
       return str
    end
    local autotunerHash = shape(self.weight) .. ';'
       .. shape(input[input_slice]) .. ';'
       .. shape(self.output[output_slice])

    local maxBufSize = 0

    -- create forwardAlgorithm descriptors
    local algType = ffi.new("cudnnConvolutionFwdAlgo_t[?]", 1)
    local algSearchMode = 'CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT'
    local algWorkspaceLimit = self.workspace_limit
       or (self.nInputPlane * self.kH * self.kW * 4) -- 4 = sizeof int/float.

    if self.fastest_mode or cudnn.fastest == true then
       algSearchMode = 'CUDNN_CONVOLUTION_FWD_PREFER_FASTEST'
    end

    if cudnn.benchmark then -- the manual auto-tuner is run
       if autotunerCache[1][autotunerHash] then
          algType[0] = autotunerCache[1][autotunerHash]
          if cudnn.verbose then
             print('Using cached benchmark for: ', autotunerHash)
          end
       else
          local perfResults = ffi.new("cudnnConvolutionFwdAlgoPerf_t[?]", 1)
          local intt = torch.IntTensor(1);
          errcheck('cudnnFindConvolutionForwardAlgorithm',
                   cudnn.getHandle(),
                   self.iDesc[0], self.weightDesc[0],
                   self.convDesc[0], self.oDesc[0],
                   1, intt:data(), perfResults)
          algType[0] = perfResults[0].algo
          autotunerCache[1][autotunerHash] = perfResults[0].algo
          if cudnn.verbose then
             print(string.format(
                      "Autotuning        Forward: Time: %3.5f Memory: %8d Algorithm: %d"
                         .. " Weight: %15s Input: %15s Output: %15s",
                      perfResults[0].time, tonumber(perfResults[0].memory),
                      tonumber(perfResults[0].algo),
                      shape(self.weight), shape(input[input_slice]),
                      shape(self.output[output_slice])))
          end
       end
    else
       errcheck('cudnnGetConvolutionForwardAlgorithm',
                cudnn.getHandle(),
                self.iDesc[0], self.weightDesc[0],
                self.convDesc[0], self.oDesc[0],
                algSearchMode, algWorkspaceLimit, algType)
    end

    algType[0] = self.fmode or algType[0]
    self.fwdAlgType = algType
    local bufSize = torch.LongTensor(1)
    errcheck('cudnnGetConvolutionForwardWorkspaceSize',
             cudnn.getHandle(),
             self.iDesc[0], self.weightDesc[0],
             self.convDesc[0], self.oDesc[0],
             algType[0], bufSize:data())
    maxBufSize = math.max(maxBufSize, bufSize[1])

    -- create backwardFilterAlgorithm descriptors
    local algType = ffi.new("cudnnConvolutionBwdFilterAlgo_t[?]", 1)
    local algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE'
    local algWorkspaceLimit = self.workspace_limit
       or (self.nInputPlane * self.kH * self.kW * 4) -- 4 = sizeof int/float.
    if self.fastest_mode  or cudnn.fastest == true then
       algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST'
    end

    if cudnn.benchmark then -- the manual auto-tuner is run
       if autotunerCache[2][autotunerHash] then
          algType[0] = autotunerCache[2][autotunerHash]
       else
          local perfResults = ffi.new("cudnnConvolutionBwdFilterAlgoPerf_t[?]", 1)
          local intt = torch.IntTensor(1);
          errcheck('cudnnFindConvolutionBackwardFilterAlgorithm',
                   cudnn.getHandle(),
                   self.iDesc[0], self.oDesc[0],
                   self.convDesc[0], self.weightDesc[0],
                   1, intt:data(), perfResults)
          algType[0] = perfResults[0].algo
          autotunerCache[2][autotunerHash] = perfResults[0].algo
          if cudnn.verbose then
             print(string.format(
                      "Autotuning backwardFilter: Time: %3.5f Memory: %8d Algorithm: %d"
                         .. " Weight: %15s Input: %15s Output: %15s",
                      perfResults[0].time, tonumber(perfResults[0].memory),
                      tonumber(perfResults[0].algo),
                      shape(self.weight), shape(input[input_slice]),
                      shape(self.output[output_slice])))
          end
       end
    else
       errcheck('cudnnGetConvolutionBackwardFilterAlgorithm',
                cudnn.getHandle(),
                self.iDesc[0], self.oDesc[0],
                self.convDesc[0], self.weightDesc[0],
                algSearchMode, algWorkspaceLimit, algType)
    end
    algType[0] = self.bwmode or algType[0]
    self.bwdFilterAlgType = algType
    local bufSize = torch.LongTensor(1)
    errcheck('cudnnGetConvolutionBackwardFilterWorkspaceSize',
             cudnn.getHandle(),
             self.iDesc[0], self.oDesc[0],
             self.convDesc[0], self.weightDesc[0],
             algType[0], bufSize:data())
    maxBufSize = math.max(maxBufSize, bufSize[1])

    -- create backwardDataAlgorithm descriptors
    local algType = ffi.new("cudnnConvolutionBwdDataAlgo_t[?]", 1)
    local algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE'
    local algWorkspaceLimit = self.workspace_limit
       or (self.nInputPlane * self.kH * self.kW * 4) -- 4 = sizeof int/float.
    if self.fastest_mode or cudnn.fastest == true then
       algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST'
    end
    if cudnn.benchmark then -- the manual auto-tuner is run
       if autotunerCache[3][autotunerHash] then
          algType[0] = autotunerCache[3][autotunerHash]
       else
          local perfResults = ffi.new("cudnnConvolutionBwdDataAlgoPerf_t[?]", 1)
          local intt = torch.IntTensor(1);
          errcheck('cudnnFindConvolutionBackwardDataAlgorithm',
                   cudnn.getHandle(),
                   self.weightDesc[0], self.oDesc[0],
                   self.convDesc[0], self.iDesc[0],
                   1, intt:data(), perfResults)
          algType[0] = perfResults[0].algo
          autotunerCache[3][autotunerHash] = perfResults[0].algo
          if cudnn.verbose then
             print(string.format(
                      "Autotuning   backwardData: Time: %3.5f Memory: %8d Algorithm: %d"
                         .. " Weight: %15s Input: %15s Output: %15s\n",
                      perfResults[0].time, tonumber(perfResults[0].memory),
                      tonumber(perfResults[0].algo),
                      shape(self.weight), shape(input[input_slice]),
                      shape(self.output[output_slice])))
          end
       end
    else
       errcheck('cudnnGetConvolutionBackwardDataAlgorithm',
                cudnn.getHandle(),
                self.weightDesc[0], self.oDesc[0],
                self.convDesc[0], self.iDesc[0],
                algSearchMode, algWorkspaceLimit, algType)
    end
    algType[0] = self.bdmode or algType[0]
    self.bwdDataAlgType = algType
    local bufSize = torch.LongTensor(1)
    errcheck('cudnnGetConvolutionBackwardDataWorkspaceSize',
             cudnn.getHandle(),
             self.weightDesc[0], self.oDesc[0],
             self.convDesc[0], self.iDesc[0],
             algType[0], bufSize:data())
    maxBufSize = math.max(maxBufSize, bufSize[1])

    self.extraBuffer = self.extraBuffer or cudnn.getSharedWorkspace()
    self.extraBufferSizeInBytes = self.extraBuffer:nElement() * 4 -- float
    if maxBufSize > self.extraBufferSizeInBytes then
       self.extraBuffer:resize(math.ceil(maxBufSize/4))
       self.extraBufferSizeInBytes = maxBufSize
    end

    -----------------------------------------------------------------------
    -- create offsets for groups
    local iH, iW = input:size(3), input:size(4)
    local kH, kW = self.kH, self.kW
    local oH, oW = oSize[3], oSize[4]
    self.input_offset = self.nInputPlane / self.groups * iH * iW
    self.output_offset = self.nOutputPlane / self.groups * oH * oW
    self.weight_offset = self.nInputPlane / self.groups * self.nOutputPlane / self.groups * kH * kW

    if not batch then
       if mydim == 5 then
          self.gradInput = self.gradInput:view(self.gradInput:size(2),
                                               self.gradInput:size(3),
                                               self.gradInput:size(4),
                                               self.gradInput:size(5))
          self.output = self.output:view(self.output:size(2),
                                         self.output:size(3),
                                         self.output:size(4),
                                         self.output:size(5))
       elseif mydim == 4 then

          self.gradInput = self.gradInput:view(self.gradInput:size(2),
                                               self.gradInput:size(3),
                                               self.gradInput:size(4))
          self.output = self.output:view(self.output:size(2),
                                         self.output:size(3),
                                         self.output:size(4))
       end
    end
    return true
end

local one = torch.FloatTensor({1});
local zero = torch.FloatTensor({0});

function ConvolutionImpl:makeContiguous(input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:typeAs(input):resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput and not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:typeAs(gradOutput):resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   return input, gradOutput
end


function ConvolutionImpl:updateOutput(input)
    -- callers of this impl methods should make sure it's contiguouos
    assert(input:isContiguous(), 'input has to be contiguous')
    if not self.weightDesc then self:resetWeightDescriptors() end
    self:createIODescriptors(input)
    for g = 0, self.groups - 1 do
        errcheck('cudnnConvolutionForward', cudnn.getHandle(),
                 one:data(),
                 self.iDesc[0], input:data() + g*self.input_offset,
                 self.weightDesc[0], self.weight:data() + g*self.weight_offset,
                 self.convDesc[0], self.fwdAlgType[0],
                 self.extraBuffer:data(), self.extraBufferSizeInBytes,
                 zero:data(),
                 self.oDesc[0], self.output:data() + g*self.output_offset);
    end

    -- add bias
    if self.bias then
        errcheck('cudnnAddTensor', cudnn.getHandle(),
                 one:data(), self.biasDesc[0], self.bias:data(),
                 one:data(), self.oDescForBias[0], self.output:data())
    end

    return self.output
end

function ConvolutionImpl:updateGradInput(input, gradOutput)
    if not self.gradInput then return end
    local mydim =  self.nDim or 4
    assert(gradOutput:dim() == mydim-1 or gradOutput:dim() == mydim,
           string.format('gradOutput has to be %sD or %sD', mydim-1, mydim))
    -- callers of this impl methods should make sure it's contiguouos
    assert(gradOutput:isContiguous(), 'gradOutput has to be contiguous')
    if not self.weightDesc then self:resetWeightDescriptors() end
    self:createIODescriptors(input)

    for g = 0,self.groups - 1 do
        errcheck('cudnnConvolutionBackwardData', cudnn.getHandle(),
                 one:data(),
                 self.weightDesc[0], self.weight:data() + g*self.weight_offset,
                 self.oDesc[0], gradOutput:data() + g*self.output_offset,
                 self.convDesc[0],
                 self.bwdDataAlgType[0],
                 self.extraBuffer:data(), self.extraBufferSizeInBytes,
                 zero:data(),
                 self.iDesc[0], self.gradInput:data() + g*self.input_offset);
    end
    return self.gradInput
end

function ConvolutionImpl:accGradParameters(input, gradOutput, scale)
    self.scaleT = self.scaleT or torch.FloatTensor(1):fill(1.0)
    -- this line forces this member to always be on CPU (needed for cudnn)
    self.scaleT = self.scaleT:float()
    scale = scale or 1.0
    self.scaleT[1] = scale
    mydim = self.nDim or 4
    assert(gradOutput:dim() == mydim-1 or gradOutput:dim() == mydim, string.format('gradOutput has to be %sD or %sD', mydim-1, mydim))
    if not self.weightDesc then self:resetWeightDescriptors() end
    self:createIODescriptors(input)

    -- gradBias
    if self.bias then
        errcheck('cudnnConvolutionBackwardBias', cudnn.getHandle(),
                 self.scaleT:data(),
                 self.oDescForBias[0], gradOutput:data(),
                 one:data(),
                 self.biasDesc[0], self.gradBias:data())
    end

    for g = 0, self.groups - 1 do
        -- gradWeight
        errcheck('cudnnConvolutionBackwardFilter', cudnn.getHandle(),
                 self.scaleT:data(),
                 self.iDesc[0], input:data() + g*self.input_offset,
                 self.oDesc[0], gradOutput:data() + g*self.output_offset,
                 self.convDesc[0],
                 self.bwdFilterAlgType[0],
                 self.extraBuffer:data(), self.extraBufferSizeInBytes,
                 one:data(),
                 self.weightDesc[0], self.gradWeight:data() + g*self.weight_offset);
    end
   return self
end

function ConvolutionImpl:clearDesc()
    self.weightDesc = nil
    self.biasDesc = nil
    self.convDesc = nil
    self.iDesc = nil
    self.oDesc = nil
    self.oDescForBias = nil
    self.algType = nil
    self.fwdAlgType = nil
    self.bwdDataAlgType = nil
    self.bwdFilterAlgType = nil
    self.extraBuffer = nil
    self.extraBufferSizeInBytes = nil
    self.scaleT = nil
    self.iSize = nil
    return self
end

function ConvolutionImpl:write(f)
    ConvolutionImpl.clearDesc(self)
    local var = {}
    for k,v in pairs(self) do
        var[k] = v
    end
    f:writeObject(var)
    return self
end

function ConvolutionImpl:clearState()
   ConvolutionImpl.clearDesc(self)
   return nn.Module.clearState(self)
end
