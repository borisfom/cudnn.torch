local ffi = require 'ffi'
local errcheck = cudnn.errcheck

local algo = {}
local autotunerCache = {}
autotunerCache['cudnnFindConvolutionForwardAlgorithmEx'] = {}
autotunerCache['cudnnFindConvolutionBackwardFilterAlgorithmEx'] = {}
autotunerCache['cudnnFindConvolutionBackwardDataAlgorithmEx'] = {}

local function setupAlgo(self, algo_t, perf_t, findAPI, getAPI, wsAPI, algSearchMode, params)
        local algType = ffi.new(algo_t, 1)
        self.extraBuffer = self.extraBuffer or cudnn.getSharedWorkspace()

        if false then -- cudnn.benchmark or cudnn.fastest then -- the manual auto-tuner is run
           local cachedAlgo = autotunerCache[findAPI][self.autotunerHash];
            if cachedAlgo then
               algType[0] = cachedAlgo
                if cudnn.verbose then
                   print('\n', findAPI, ' using cached algo = ' , algType[0] , ' for: ', self.autotunerHash)
                end
            else
                local perfResults = ffi.new(perf_t, 1)
                local intt = torch.IntTensor(1)
                errcheck(findAPI,
                         cudnn.getHandle(),
                         params[1], params[2], params[3], params[4], params[5], params[6], params[7],
                         1, intt:data(), perfResults, self.extraBuffer:data(), self.extraBuffer:nElement() * self.extraBuffer.elementSize())
                algType[0] = perfResults[0].algo
                autotunerCache[findAPI][self.autotunerHash] = perfResults[0].algo
                if cudnn.verbose then
                    print(string.format(
                              "\n" .. findAPI .. " Time: %3.5f Memory: %8d Algorithm: %d"
                                  .. " hash: %45s",
                              perfResults[0].time, tonumber(perfResults[0].memory),
                              tonumber(perfResults[0].algo), self.autotunerHash ))

                end
            end
        else

           local algWorkspaceLimit = self.workspace_limit
              or (self.nInputPlane * self.kH * self.kW * self.weight.elementSize())

            errcheck(getAPI,
                     cudnn.getHandle(),
                     params[1], params[3], params[5], params[6],
                     algSearchMode, algWorkspaceLimit, algType)
                if cudnn.verbose then
                   print(string.format(
                     "\n" .. getAPI .. " Limit: %d Algorithm: %d",
                     tonumber(algWorkspaceLimit),
                     tonumber(algType[0])))
                end
        end


        if algSearchMode ~= CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE and algSearchMode ~= CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE then
           local bufSize = torch.LongTensor(1)
           errcheck(wsAPI,
                    cudnn.getHandle(),
                    params[1], params[3], params[5], params[6],
                    algType[0], bufSize:data())

           local extraBufferSizeInBytes = self.extraBuffer:nElement() * self.extraBuffer.elementSize()

           if cudnn.verbose then
              print(string.format(
                       "\n" .. wsAPI .. " returned bufSize: %d, current extraBufferSizeInBytes: %d, %d elements",
                       tonumber(bufSize[1]), tonumber(extraBufferSizeInBytes), tonumber(self.extraBuffer:nElement())))
           end

           if extraBufferSizeInBytes < bufSize[1] then
              self.extraBuffer:resize(math.ceil(bufSize[1]/self.extraBuffer.elementSize()))
           end
        end
        return algType[0]
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
                'cudnnFindConvolutionForwardAlgorithmEx', 'cudnnGetConvolutionForwardAlgorithm',
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
                'cudnnFindConvolutionBackwardFilterAlgorithmEx', 'cudnnGetConvolutionBackwardFilterAlgorithm',
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
                'cudnnFindConvolutionBackwardDataAlgorithmEx', 'cudnnGetConvolutionBackwardDataAlgorithm',
                'cudnnGetConvolutionBackwardDataWorkspaceSize', algSearchMode, params)
end

return algo
