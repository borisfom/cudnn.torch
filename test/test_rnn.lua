require 'cudnn'
require 'cunn'
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

local cudnntest = torch.TestSuite()
local mytester

local tolerance = 70


function cudnntest.testRNNRELU()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 2
    local mode = 'CUDNN_RNN_RELU'
    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, mode)

    -- Checksums to check against are retrieved from cudnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 1.315793E+06, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 1.315212E+05, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 6.676003E+01, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 6.425067E+01, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 1.453750E+09, tolerance, 'checkSum with reference for localSumdw failed')
end

function cudnntest.testRNNTANH()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 2
    local mode = 'CUDNN_RNN_TANH'
    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, mode)

    -- Checksums to check against are retrieved from cudnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 6.319591E+05, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 6.319605E+04, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 4.501830E+00, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 4.489546E+00, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 5.012598E+07, tolerance, 'checkSum with reference for localSumdw failed')
end

function cudnntest.testRNNLSTM()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 8
    local mode = 'CUDNN_LSTM'
    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, mode)

    -- Checksums to check against are retrieved from cudnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 5.749536E+05, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumc, 4.365091E+05, tolerance, 'checkSum with reference for localSumc failed')
    mytester:assertalmosteq(checkSums.localSumh, 5.774818E+04, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 3.842206E+02, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdc, 9.323785E+03, tolerance, 'checkSum with reference for localSumdc failed')
    mytester:assertalmosteq(checkSums.localSumdh, 1.182566E+01, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 4.313461E+08, tolerance, 'checkSum with reference for localSumdw failed')
end

function cudnntest.testRNNGRU()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 6
    local mode = 'CUDNN_GRU'
    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, mode)

    -- Checksums to check against are retrieved from cudnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 6.358978E+05, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 6.281680E+04, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 6.296622E+00, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 2.289960E+05, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 5.397419E+07, tolerance, 'checkSum with reference for localSumdw failed')
end

--[[
-- Method gets Checksums of RNN to compare with ref Checksums in cudnn RNN C sample.
-- ]]
function getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, mode)

    local input = torch.CudaTensor(seqLength, miniBatch, hiddenSize):fill(1) -- Input initialised to 1s.
    local rnn = cudnn.RNN(hiddenSize, hiddenSize, numberOfLayers)
    rnn.mode = mode -- Set the mode (GRU/LSTM/ReLU/tanh
    rnn:reset()
    rnn:resetWeightDescriptor()

    -- Matrices are initialised to 1 / matrixSize, biases to 1.
    for layer = 0, numberOfLayers - 1 do
        for layerId = 0, numberOfLinearLayers - 1 do
            local linLayerMatDesc = rnn:createFilterDescriptors(1)
            local matrixPointer = ffi.new("float*[1]")
            errcheck('cudnnGetRNNLinLayerMatrixParams',
                cudnn.getHandle(),
                rnn.rnnDesc[0],
                layer,
                rnn.xDescs,
                rnn.wDesc[0],
                rnn.weight:data(),
                layerId,
                linLayerMatDesc[0],
                ffi.cast("void**", matrixPointer))

            local dataType = 'CUDNN_DATA_FLOAT'
            local format = 'CUDNN_TENSOR_NCHW'
            local nbDims = torch.IntTensor(1)
            local filterDimA = torch.IntTensor({ 1, 1, 1 })

            errcheck('cudnnGetFilterNdDescriptor',
                linLayerMatDesc[0],
                3,
                ffi.cast("cudnnDataType_t*", dataType),
                ffi.cast("cudnnDataType_t*", format),
                nbDims:data(),
                filterDimA:data())

            local offset = matrixPointer[0] - rnn.weight:data()
            local weightTensor = torch.CudaTensor(rnn.weight:storage(), offset + 1, filterDimA[1] * filterDimA[2] * filterDimA[3])
            weightTensor:fill(1.0 / (filterDimA[1] * filterDimA[2] * filterDimA[3]))

            local linLayerBiasDesc = rnn:createFilterDescriptors(1)
            local biasPointer = ffi.new("float*[1]")
            errcheck('cudnnGetRNNLinLayerBiasParams',
                cudnn.getHandle(),
                rnn.rnnDesc[0],
                layer,
                rnn.xDescs,
                rnn.wDesc[0],
                rnn.weight:data(),
                layerId,
                linLayerBiasDesc[0],
                ffi.cast("void**", biasPointer))

            local dataType = 'CUDNN_DATA_FLOAT'
            local format = 'CUDNN_TENSOR_NCHW'
            local nbDims = torch.IntTensor(1)
            local filterDimA = torch.IntTensor({ 1, 1, 1 })

            errcheck('cudnnGetFilterNdDescriptor',
                linLayerBiasDesc[0],
                3,
                ffi.cast("cudnnDataType_t*", dataType),
                ffi.cast("cudnnDataType_t*", format),
                nbDims:data(),
                filterDimA:data())

            local offset = biasPointer[0] - rnn.weight:data()
            local biasTensor = torch.CudaTensor(rnn.weight:storage(), offset + 1, filterDimA[1] * filterDimA[2] * filterDimA[3])
            biasTensor:fill(1)
        end
    end
    -- Set hx/cx/dhy/dcy data to 1s.
    rnn.hiddenInput = torch.CudaTensor(numberOfLayers, miniBatch, hiddenSize):fill(1)
    rnn.cellInput = torch.CudaTensor(numberOfLayers, miniBatch, hiddenSize):fill(1)
    rnn.gradHiddenOutput = torch.CudaTensor(numberOfLayers, miniBatch, hiddenSize):fill(1)
    rnn.gradCellOutput = torch.CudaTensor(numberOfLayers, miniBatch, hiddenSize):fill(1)

    local testOutputi = rnn:forward(input)
    -- gradInput set to 1s.
    local gradInput = torch.CudaTensor(seqLength, miniBatch, hiddenSize):fill(1)
    rnn:backward(input, gradInput)

    -- Sum up all values for each.
    local localSumi = 0
    local localSumh = 0
    local localSumc = 0
    for m = 1, seqLength do
        for j = 1, miniBatch do
            for i = 1, hiddenSize do
                localSumi = localSumi + testOutputi[m][j][i]
            end
        end
    end

    for m = 1, numberOfLayers do
        for j = 1, miniBatch do
            for i = 1, hiddenSize do
                localSumh = localSumh + rnn.hiddenOutput[m][j][i]
                localSumc = localSumc + rnn.cellOutput[m][j][i]
            end
        end
    end

    local localSumdi = 0
    local localSumdh = 0
    local localSumdc = 0
    for m = 1, seqLength do
        for j = 1, miniBatch do
            for i = 1, hiddenSize do
                localSumdi = localSumdi + rnn.gradInput[m][j][i]
            end
        end
    end
    for m = 1, numberOfLayers do
        for j = 1, miniBatch do
            for i = 1, hiddenSize do
                localSumdh = localSumdh + rnn.gradHiddenInput[m][j][i]
                localSumdc = localSumdc + rnn.gradCellInput[m][j][i]
            end
        end
    end

    local localSumdw = 0
    for m = 1, rnn.gradWeight:size(1) do
        localSumdw = localSumdw + rnn.gradWeight[m]
    end

    local checkSums = {
        localSumi = localSumi,
        localSumh = localSumh,
        localSumc = localSumc,
        localSumdi = localSumdi,
        localSumdh = localSumdh,
        localSumdc = localSumdc,
        localSumdw = localSumdw
    }
    return checkSums
end

mytester = torch.Tester()
mytester:add(cudnntest)
mytester:run()