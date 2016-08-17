local ffi = require 'ffi'

find = {}
find.__index = find

function find.create(id)
   local finder = {}
   setmetatable(finder,find)
   finder.id = id
   finder.autotunerCache = {{}, {}, {}}
   finder.calculatedWorkspaceSize  = 0
   finder.useDefaultWorkspaceSize = true
   finder.useMaxWorkspaceSize = false
   finder.useCalculatedWorkspaceSize = false
   finder.useFindEx = cudnn.useFindEx and (cudnn.benchmark or cudnn.fastest)
   finder.iteration = 0
   return finder
end

local finders = nil

function find.reset()
   cudnn.adjustSharedWorkspaceSize(0)
   finders = {}
end

function find.get()
   local device = cutorch.getDevice()
   local it = finders[device]
   if not it then
      it = find.create(device)
      finders[device] = it
   end
   return it
end

function find:lookup(layer, findAPI_idx)
   return  self.autotunerCache[findAPI_idx][layer.autotunerHash]
end

function find:store(layer, findAPI_idx, cachedAlgo)
   self.autotunerCache[findAPI_idx][layer.autotunerHash] = cachedAlgo
end

function find:setMaxWorkspaceSize(reserve, fraction)
   local freeMemory, totalMemory = cutorch.getMemoryUsage(self.id)
   reserve = reserve or 0
   fraction = fraction or 0.8
   cudnn.adjustSharedWorkspaceSize((freeMemory -reserve) * fraction)
   self.useMaxWorkspaceSize = true
   self.useDefaultWorkspaceSize = false
   self.useCalculatedWorkspaceSize = false
end

function find:setCalculatedWorkspaceSize(greater)
   cudnn.adjustSharedWorkspaceSize(self.calculatedWorkspaceSize, greater)
   self.useMaxWorkspaceSize = false
   self.useDefaultWorkspaceSize = false
   self.useCalculatedWorkspaceSize = true
end

function find:registerWorkspaceSize(size)
   if size > self.calculatedWorkspaceSize then
      self.calculatedWorkspaceSize = size
   end
   if not self.useFindEx then
      self:setCalculatedWorkspaceSize(true) -- set immediately like we did before
   end
end

function find:updateWorkspaceModes(layer)
   if self.useCalculatedWorkspaceSize then
      return
   end
   -- find last iteration
   local max_iter = 0
   for k,v in pairs(layer.iteration) do
      if v > max_iter then max_iter = v end
   end
   if (self.iteration < max_iter) then
      self.iteration = max_iter
      -- iteration changed, advance state machine
      if self.useMaxWorkspaceSize then
         self:setCalculatedWorkspaceSize()
      end
      if self.useDefaultWorkspaceSize then
         if self.useFindEx then
            local reserve = layer.weight:nElement()*layer.weight:elementSize()*4
            self:setMaxWorkspaceSize(reserve)
         else
            self:setCalculatedWorkspaceSize(true)
         end
      end
   end
end


local findNoExAlgos = {'cudnnFindConvolutionForwardAlgorithm', 'cudnnFindConvolutionBackwardFilterAlgorithm', 'cudnnFindConvolutionBackwardDataAlgorithm'}
local findExAlgos = {'cudnnFindConvolutionForwardAlgorithmEx', 'cudnnFindConvolutionBackwardFilterAlgorithmEx', 'cudnnFindConvolutionBackwardDataAlgorithmEx'}

local function call(self, f, ...)
   local status = cudnn.call(f, ...)
   if cudnn.verbose and status ~= ffi.C.CUDNN_STATUS_SUCCESS then
      print(f .. " failed for sizes: " .. self.autotunerHash)
   end
   return status
end
find.call = call

local function errcheck(self, f, ...)
   local status = call(self, f, ...)
   if status ~= ffi.C.CUDNN_STATUS_SUCCESS then
      local str = ffi.string(cudnn.C.cudnnGetErrorString(status))
      error('Error in CuDNN: ' .. str .. ' ('..f..')')
   end
end
find.errcheck = errcheck

function find:setupAlgo(layer, algo_t, perf_t, findAPI_idx, getAPI, wsAPI, algSearchMode, params)
        local findAPI = findNoExAlgos[findAPI_idx]
        local findExAPI = findExAlgos[findAPI_idx]

        local cacheHit = '[found in cache]'

        if not layer.iteration then layer.iteration = {0,0,0} end
        layer.iteration[findAPI_idx] = layer.iteration[findAPI_idx] + 1
        local targetParam = params[7]
        local curFindAPI = findAPI
        if self.useFindEx then
           -- do the clone before workspace size calculation
           targetParam=targetParam:clone()
           curFindAPI = findExAPI
        end
        -- advance state machine
        self:updateWorkspaceModes(layer)
        layer.extraBuffer = cudnn.getSharedWorkspace()
        local extraBufferSizeInBytes = 0

        local cachedAlgo =  self:lookup(layer, findAPI_idx)

        -- do not trust the cache until workspace sizes are set up
        if not cachedAlgo or not self.useCalculatedWorkspaceSize then
           cacheHit = ''
           cachedAlgo = {}
           local perfResults = ffi.new(perf_t, 1)

           if cudnn.benchmark or cudnn.fastest then -- the manual auto-tuner is run
              local intt = torch.IntTensor(1)
              if cudnn.useFindEx then
                 extraBufferSizeInBytes = layer.extraBuffer:size()*layer.extraBuffer:elementSize()
                 errcheck(layer, findExAPI,
                          cudnn.getHandle(),
                          params[1], params[2]:data(), params[3], params[4]:data(), params[5], params[6], targetParam:data(),
                          1, intt:data(), perfResults, layer.extraBuffer:data(), extraBufferSizeInBytes)
              else
                 curFindAPI = findAPI
                 errcheck(layer, findAPI,
                          cudnn.getHandle(),
                          params[1], params[3], params[5], params[6],
                          1, intt:data(), perfResults)
              end

              cachedAlgo.algo = tonumber(perfResults[0].algo)
              cachedAlgo.memory = tonumber(perfResults[0].memory)
              --- todo: use fallback if status ~= 0
              cachedAlgo.status = tonumber(perfResults[0].status)

              if cudnn.verbose then
                 print(string.format(
                          "\n" .. curFindAPI .. " algo: %d (status: %d), memory: %8d"
                             .. " hash: %45s " .. cacheHit,
                          cachedAlgo.algo,  cachedAlgo.status, cachedAlgo.memory, layer.autotunerHash))
              end
           else
              curFindAPI=getAPI
              local algWorkspaceLimit = layer.workspace_limit
                 or (layer.nInputPlane * layer.kH * layer.kW * layer.weight.elementSize())

              local algType = ffi.new(algo_t, 1)
              errcheck(layer, getAPI,
                       cudnn.getHandle(),
                       params[1], params[3], params[5], params[6],
                       algSearchMode, algWorkspaceLimit, algType)

              if cudnn.verbose then
                 print(string.format(
                          "\n" .. getAPI .. ": %d (ws limit: %d) mode = %s",
                          tonumber(algType[0]),
                          algWorkspaceLimit,
                          algSearchMode))
              end

              cachedAlgo.algo = tonumber(algType[0])
              local bufSize = torch.LongTensor(1)
              errcheck(layer, wsAPI,
                       cudnn.getHandle(),
                       params[1], params[3], params[5], params[6],
                       algType[0], bufSize:data())
              if cudnn.verbose then
                 print(string.format(
                          "\n" .. wsAPI  .. ": bufSize: %d, current ws: %d",
                          tonumber(bufSize[1]), tonumber(extraBufferSizeInBytes)))
              end
              cachedAlgo.memory = tonumber(bufSize[1])
              cachedAlgo.status = 0
           end
           -- record algo, memory in cache
           self:store(layer, findAPI_idx, cachedAlgo)
        end

        if cachedAlgo.status == 0 then
           -- memorize our own ws size and update global
           self:registerWorkspaceSize(cachedAlgo.memory)
        else
           -- todo: fallback (convert layer to nn ?)
        end

        if find.verbose then
           print(string.format(
                    "\n" .. curFindAPI  .. ": %d Workspace: %8d (current ws size %d)  hash: %45s" .. cacheHit,
                    tonumber(cachedAlgo.algo), tonumber(cachedAlgo.memory), layer.extraBuffer:size()*layer.extraBuffer:elementSize(), layer.autotunerHash))
        end

        return cachedAlgo.algo
end

function find.prepare(layer, input_slice, output_slice)
   local function shape(x)
      return table.concat(x:size():totable(),'x')
   end
   layer.autotunerHash = shape(layer.weight) .. ';'
      .. shape(input_slice) .. ';'
      .. shape(output_slice) .. "[" .. layer.padH .. ":" .. layer.padW .. ']'

   layer.fwdAlgType = nil
   layer.bwdDataAlgType = nil
   layer.bwdFilterAlgType = nil
   layer.input_slice = input_slice
   layer.output_slice = output_slice
end

function find.forwardAlgorithm(layer, params)
   local algSearchMode
   if layer.fastest_mode  or cudnn.benchmark == true or cudnn.fastest == true then
      algSearchMode = 'CUDNN_CONVOLUTION_FWD_PREFER_FASTEST'
   else
      algSearchMode = 'CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT'
   end
   params = params or { layer.iDesc[0], layer.input_slice, layer.weightDesc[0], layer.weight, layer.convDesc[0], layer.oDesc[0], layer.output_slice}
   -- supply a temporary for findEx
   layer.fwdAlgType =  find.get():setupAlgo(layer,"cudnnConvolutionFwdAlgo_t[?]", "cudnnConvolutionFwdAlgoPerf_t[?]",
                                            1, 'cudnnGetConvolutionForwardAlgorithm',
                                            'cudnnGetConvolutionForwardWorkspaceSize', algSearchMode, params)
   if layer.fmode then
      -- is it obsolete ?
   end
end

function find.backwardFilterAlgorithm(layer, params)
   local algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE'
   if layer.fastest_mode  or cudnn.fastest == true then
      algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST'
   end
   params = params or { layer.iDesc[0], layer.input_slice, layer.oDesc[0], layer.output_slice, layer.convDesc[0], layer.weightDesc[0], layer.weight}
   -- supply a temporary for findEx
   layer.bwdFilterAlgType = find.get():setupAlgo(layer,"cudnnConvolutionBwdFilterAlgo_t[?]", "cudnnConvolutionBwdFilterAlgoPerf_t[?]",
                                                 2, 'cudnnGetConvolutionBackwardFilterAlgorithm',
                                                 'cudnnGetConvolutionBackwardFilterWorkspaceSize', algSearchMode,
                                                 params)
   --   if layer.bwmode then ?
end

function find.backwardDataAlgorithm(layer, params)
   local algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE'
   if layer.fastest_mode  or cudnn.fastest == true then
      algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST'
   end
   params =  params or { layer.weightDesc[0], layer.weight, layer.oDesc[0], layer.output_slice, layer.convDesc[0], layer.iDesc[0], layer.input_slice }
   -- supply a temporary for findEx
   layer.bwdDataAlgType = find.get():setupAlgo(layer,"cudnnConvolutionBwdDataAlgo_t[?]", "cudnnConvolutionBwdDataAlgoPerf_t[?]",
                                               3, 'cudnnGetConvolutionBackwardDataAlgorithm',
                                               'cudnnGetConvolutionBackwardDataWorkspaceSize', algSearchMode, params)
   -- if layer.bdmode then ?
end

find.reset()
return find
