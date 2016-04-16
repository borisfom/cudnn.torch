--[[
--  Tests the implementation of RNN binding using the cudnn v5 library. Cross-check the checksums with cudnn reference
--  sample checksums.
-- ]]

require 'cudnn'
require 'cunn'
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

local cudnntest = torch.TestSuite()
local mytester

local tolerance = 300

function cudnntest.testRNNRELU()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 2
    local rnn = cudnn.RNNReLU(hiddenSize, hiddenSize, numberOfLayers)
    rnn.mode = 'CUDNN_RNN_RELU'
    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn)

    -- Checksums to check against are retrieved from cudnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 1.315793E+06, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 1.315212E+05, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 6.676003E+01, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 6.425067E+01, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 1.453750E+09, tolerance, 'checkSum with reference for localSumdw failed')
end

function cudnntest.testRNNBatchFirst()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 2
    local batchFirst = true
    local rnn = cudnn.RNNReLU(hiddenSize, hiddenSize, numberOfLayers, batchFirst)
    rnn.mode = 'CUDNN_RNN_RELU'
    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn, batchFirst)

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
    local rnn = cudnn.RNNTanh(hiddenSize, hiddenSize, numberOfLayers)
    rnn.mode = 'CUDNN_RNN_TANH'
    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn)

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
    local rnn = cudnn.LSTM(hiddenSize, hiddenSize, numberOfLayers)
    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn)

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
    local rnn = cudnn.GRU(hiddenSize, hiddenSize, numberOfLayers)
    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn)
    -- Checksums to check against are retrieved from cudnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 6.358978E+05, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 6.281680E+04, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 6.296622E+00, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 2.289960E+05, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 5.397419E+07, tolerance, 'checkSum with reference for localSumdw failed')
end

function cudnntest.testBiDirectionalRELURNN()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 2
    local nbDirections = 2
    local batchFirst = false
    local rnn = cudnn.RNN(hiddenSize, hiddenSize, numberOfLayers)
    rnn.bidirectional = 'CUDNN_BIDIRECTIONAL'
    rnn.mode = 'CUDNN_RNN_RELU'
    rnn.numDirections = 2

    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn, batchFirst, nbDirections)
    -- Checksums to check against are retrieved from cudnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 1.388634E+01, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 1.288997E+01, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 1.288729E+01, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 1.279004E+01, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 7.061081E+07, tolerance, 'checkSum with reference for localSumdw failed')
end

function cudnntest.testBiDirectionalTANHRNN()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 2
    local nbDirections = 2
    local batchFirst = false
    local rnn = cudnn.RNN(hiddenSize, hiddenSize, numberOfLayers)
    rnn.bidirectional = 'CUDNN_BIDIRECTIONAL'
    rnn.mode = 'CUDNN_RNN_TANH'
    rnn.numDirections = 2

    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn, batchFirst, nbDirections)
    -- Checksums to check against are retrieved from cudnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 1.388634E+01, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 1.288997E+01, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 1.288729E+01, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 1.279004E+01, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 7.061081E+07, tolerance, 'checkSum with reference for localSumdw failed')
end

function cudnntest.testBiDirectionalLSTMRNN()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 8
    local nbDirections = 2
    local batchFirst = false
    local rnn = cudnn.BLSTM(hiddenSize, hiddenSize, numberOfLayers)

    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn, batchFirst, nbDirections)
    -- Checksums to check against are retrieved from cudnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 3.134097E+04, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumc, 3.845626E+00, tolerance, 'checkSum with reference for localSumc failed')
    mytester:assertalmosteq(checkSums.localSumh, 1.922855E+00, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 4.794993E+00, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdc, 2.870925E+04, tolerance, 'checkSum with reference for localSumdc failed')
    mytester:assertalmosteq(checkSums.localSumdh, 2.468645E+00, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 1.121568E+08, tolerance, 'checkSum with reference for localSumdw failed')
end

function cudnntest.testBiDirectionalGRURNN()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 6
    local nbDirections = 2
    local batchFirst = false
    local rnn = cudnn.RNN(hiddenSize, hiddenSize, numberOfLayers)
    rnn.bidirectional = 'CUDNN_BIDIRECTIONAL'
    rnn.mode = 'CUDNN_GRU'
    rnn.numDirections = 2

    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn, batchFirst, nbDirections)
    -- Checksums to check against are retrieved from cudnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 6.555183E+04, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 5.830924E+00, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 4.271801E+00, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 6.555744E+04, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 1.701796E+08, tolerance, 'checkSum with reference for localSumdw failed')
end

--[[
-- Method gets Checksums of RNN to compare with ref Checksums in cudnn RNN C sample.
-- ]]
function getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn, batchFirst, nbDirections)
    local biDirectionalScale = nbDirections or 1
    -- Reset the rnn and weight descriptor (since we are manually setting values for matrix/bias.
    rnn:reset()
    rnn:resetWeightDescriptor()
    local input
    if (batchFirst) then
        input = torch.CudaTensor(miniBatch, seqLength, hiddenSize):fill(1)
    else
        input = torch.CudaTensor(seqLength, miniBatch, hiddenSize):fill(1) -- Input initialised to 1s.
    end
    if (biDirectionalScale == 2) then
        rnn.weight:fill(1 / rnn.weight:size(1))
    else
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

                local minDim = 3
                local filterDimA = torch.ones(minDim):int()
                errcheck('cudnnGetFilterNdDescriptor',
                    linLayerMatDesc[0],
                    minDim,
                    ffi.cast("cudnnDataType_t*", dataType),
                    ffi.cast("cudnnDataType_t*", format),
                    nbDims:data(),
                    filterDimA:data())

                local offset = matrixPointer[0] - rnn.weight:data()
                local weightTensor = torch.CudaTensor(rnn.weight:storage(), offset + 1, filterDimA:prod())
                weightTensor:fill(1.0 / filterDimA:prod())

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
                local filterDimA = torch.ones(minDim):int()

                errcheck('cudnnGetFilterNdDescriptor',
                    linLayerBiasDesc[0],
                    minDim,
                    ffi.cast("cudnnDataType_t*", dataType),
                    ffi.cast("cudnnDataType_t*", format),
                    nbDims:data(),
                    filterDimA:data())

                local offset = biasPointer[0] - rnn.weight:data()
                local biasTensor = torch.CudaTensor(rnn.weight:storage(), offset + 1, filterDimA:prod())
                biasTensor:fill(1)
            end
        end
    end
    -- Set hx/cx/dhy/dcy data to 1s.
    rnn.hiddenInput = torch.CudaTensor(numberOfLayers * biDirectionalScale, miniBatch, hiddenSize):fill(1)
    rnn.cellInput = torch.CudaTensor(numberOfLayers * biDirectionalScale, miniBatch, hiddenSize):fill(1)
    rnn.gradHiddenOutput = torch.CudaTensor(numberOfLayers * biDirectionalScale, miniBatch, hiddenSize):fill(1)
    rnn.gradCellOutput = torch.CudaTensor(numberOfLayers * biDirectionalScale, miniBatch, hiddenSize):fill(1)
    local testOutputi = rnn:forward(input)
    -- gradInput set to 1s.
    local gradInput
    if(batchFirst) then
        gradInput = torch.CudaTensor(miniBatch, seqLength, hiddenSize * biDirectionalScale):fill(1)
    else
        gradInput = torch.CudaTensor(seqLength, miniBatch, hiddenSize * biDirectionalScale):fill(1)
    end
    rnn:backward(input, gradInput)

    -- Sum up all values for each.
    local localSumi = 0
    local localSumh = 0
    local localSumc = 0

    local mLength
    local jLength
    if(batchFirst) then
        mLength = miniBatch
        jLength = seqLength
    else
        mLength = seqLength
        jLength = miniBatch
    end

    for m = 1, mLength do
        for j = 1, jLength do
            for i = 1, hiddenSize * biDirectionalScale do
                localSumi = localSumi + testOutputi[m][j][i]
            end
        end
    end

    for m = 1, numberOfLayers * biDirectionalScale do
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
    for m = 1, mLength do
        for j = 1, jLength do
            for i = 1, hiddenSize do
                localSumdi = localSumdi + rnn.gradInput[m][j][i]
            end
        end
    end
    for m = 1, numberOfLayers * biDirectionalScale do
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