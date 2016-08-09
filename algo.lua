local ffi = require 'ffi'
local errcheck = cudnn.errcheck

local algo = {}

local findNoExAlgos = {'cudnnFindConvolutionForwardAlgorithm', 'cudnnFindConvolutionBackwardFilterAlgorithm', 'cudnnFindConvolutionBackwardDataAlgorithm'}
local findExAlgos = {'cudnnFindConvolutionForwardAlgorithmEx', 'cudnnFindConvolutionBackwardFilterAlgorithmEx', 'cudnnFindConvolutionBackwardDataAlgorithmEx'}

local autotunerCache = {{}, {}, {}}
local findAlgos = nil

local function setupAlgo(self, algo_t, perf_t, findAPI_idx, getAPI, wsAPI, algSearchMode, params)

        local function initCache(useEx)
           if useEx then
              findAlgos = findExAlgos
           else
              findAlgos = findNoExAlgos
           end
        end

        -- recheck if cudnn.useFindEx was reset
        initCache(cudnn.useFindEx)
        local findAPI = findAlgos[findAPI_idx]
        self.extraBuffer = self.extraBuffer or cudnn.getSharedWorkspace()
        local extraBufferSizeInBytes = self.extraBuffer:nElement() * self.extraBuffer.elementSize()
        local cachedAlgo = autotunerCache[findAPI_idx][self.autotunerHash]
        local cacheHit = '[found in cache]'
        if not cachedAlgo then
           cacheHit = ''
           cachedAlgo = {}
           local perfResults = ffi.new(perf_t, 1)
           if cudnn.benchmark or cudnn.fastest then -- the manual auto-tuner is run
              local intt = torch.IntTensor(1)
              if cudnn.useFindEx then
                 errcheck(findAPI,
                          cudnn.getHandle(),
                          params[1], params[2], params[3], params[4], params[5], params[6], params[7],
                          1, intt:data(), perfResults, self.extraBuffer:data(), self.extraBuffer:nElement() * self.extraBuffer.elementSize())

              else
                 errcheck(findAPI,
                          cudnn.getHandle(),
                          params[1], params[3], params[5], params[6],
                          1, intt:data(), perfResults)
              end

              if status ~= ffi.C.CUDNN_STATUS_SUCCESS then
                 cachedAlgo.algo =0
                 cachedAlgo.memory = 0
                 cachedAlgo.time = 0
              else
                 cachedAlgo.algo = perfResults[0].algo
                 cachedAlgo.memory = perfResults[0].memory
                 cachedAlgo.time = perfResults[0].time
              end
              cachedAlgo.status = perfResults[0].status

              if cudnn.verbose then
                 print(string.format(
                          "\n" .. findAPI .. " Time: %3.5f Memory: %8d Algorithm: %d(out of %d, status: %d)"
                             .. " hash: %45s",
                          cachedAlgo.time, tonumber(cachedAlgo.memory),
                          tonumber(cachedAlgo.algo), intt[1], tonumber(cachedAlgo.status), self.autotunerHash ))

              end
           else
              findAPI=getAPI
              local algType = ffi.new(algo_t, 1)
              local algWorkspaceLimit = self.workspace_limit
                 or (self.nInputPlane * self.kH * self.kW * self.weight.elementSize())
              local status = cudnn.call(getAPI,
                                        cudnn.getHandle(),
                                        params[1], params[3], params[5], params[6],
                                        algSearchMode, algWorkspaceLimit, algType)


              if cudnn.verbose then
                 print(string.format(
                          "\n" .. getAPI .. ": %d (ws limit: %d) mode = %s status=%d",
                          tonumber(algType[0]),
                          algWorkspaceLimit,
                          algSearchMode, tonumber(status)))
              end

              if status ~= ffi.C.CUDNN_STATUS_SUCCESS then
                 cachedAlgo.algo =0
                 cachedAlgo.memory = 0
              else
                 cachedAlgo.algo = algType[0]
                 local bufSize = torch.LongTensor(1)
                 status = cudnn.call(wsAPI,
                                     cudnn.getHandle(),
                                     params[1], params[3], params[5], params[6],
                                     algType[0], bufSize:data())
                 if cudnn.verbose then
                    print(string.format(
                             "\n" .. wsAPI  .. ": bufSize: %d, current ws: %d, status: %d",
                             tonumber(bufSize[1]), tonumber(extraBufferSizeInBytes), tonumber(status)))
                 end
                 if status ~= ffi.C.CUDNN_STATUS_SUCCESS then
                    cachedAlgo.memory = 0
                 else
                    cachedAlgo.memory = bufSize[1]
                 end
              end
           end
           cachedAlgo.status = perfResults[0].status

           autotunerCache[findAPI_idx][self.autotunerHash] = {}
           local cachedAlgoRef = autotunerCache[findAPI_idx][self.autotunerHash]
           cachedAlgoRef.algo = cachedAlgo.algo
           cachedAlgoRef.memory = cachedAlgo.memory
        end

        if extraBufferSizeInBytes < cachedAlgo.memory then
           self.extraBuffer:resize((tonumber(cachedAlgo.memory)+self.extraBuffer.elementSize())/self.extraBuffer.elementSize())
        end

        if cudnn.verbose then
           print(string.format(
                    "\n" .. findAPI  .. ": %d Workspace: %8d  hash: %45s" .. cacheHit,
                    tonumber(cachedAlgo.algo), tonumber(cachedAlgo.memory), self.autotunerHash))
        end
        return cachedAlgo.algo
end

function algo.prepareHash(self, input_slice, output_slice)
   local function shape(x)
      return table.concat(x:size():totable(),'x')
   end
   self.autotunerHash = shape(self.weight) .. ';'
      .. shape(input_slice) .. ';'
      .. shape(output_slice)

   self.fwdAlgType = nil
   self.bwdDataAlgType = nil
   self.bwdFilterAlgType = nil
   self.input_slice = input_slice
   self.output_slice = output_slice
end

function algo.setupForwardAlgorithm(self, params)
   local algSearchMode
   if self.fastest_mode  or cudnn.benchmark == true or cudnn.fastest == true then
      algSearchMode = 'CUDNN_CONVOLUTION_FWD_PREFER_FASTEST'
   else
      algSearchMode = 'CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT'
   end

   params = params or { self.iDesc[0], self.input_slice:data(), self.weightDesc[0], self.weight:data(), self.convDesc[0], self.oDesc[0], self.output_slice:data() }
   self.fwdAlgType = self.fmode or
      setupAlgo(self,"cudnnConvolutionFwdAlgo_t[?]", "cudnnConvolutionFwdAlgoPerf_t[?]",
                1, 'cudnnGetConvolutionForwardAlgorithm',
                'cudnnGetConvolutionForwardWorkspaceSize', algSearchMode, params)
end

function algo.setupBackwardFilterAlgorithm(self, params)
   local algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE'
   if self.fastest_mode  or cudnn.fastest == true then
      algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST'
   end
   params = params or { self.iDesc[0], self.input_slice:data(), self.oDesc[0], self.output_slice:data(), self.convDesc[0], self.weightDesc[0], self.weight:data() }
   self.bwdFilterAlgType = self.bwmode or
      setupAlgo(self,"cudnnConvolutionBwdFilterAlgo_t[?]", "cudnnConvolutionBwdFilterAlgoPerf_t[?]",
                2, 'cudnnGetConvolutionBackwardFilterAlgorithm',
                'cudnnGetConvolutionBackwardFilterWorkspaceSize', algSearchMode,
                params)
end

function algo.setupBackwardDataAlgorithm(self, params)
   local algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE'
   if self.fastest_mode  or cudnn.fastest == true then
      algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST'
   end
   params =  params or { self.weightDesc[0], self.weight:data(), self.oDesc[0], self.output_slice:data(), self.convDesc[0], self.iDesc[0], self.input_slice:data() }
   self.bwdDataAlgType = self.bdmode or
      setupAlgo(self,"cudnnConvolutionBwdDataAlgo_t[?]", "cudnnConvolutionBwdDataAlgoPerf_t[?]",
                3, 'cudnnGetConvolutionBackwardDataAlgorithm',
                'cudnnGetConvolutionBackwardDataWorkspaceSize', algSearchMode, params)
end

return algo
