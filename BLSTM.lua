local BLSTM, parent = torch.class('cudnn.BLSTM', 'cudnn.RNN')
local errcheck = cudnn.errcheck

function BLSTM:__init(inputSize, hiddenSize, numLayers)
    parent.__init(self,inputSize, hiddenSize, numLayers)

    self.datatype = 'CUDNN_DATA_FLOAT'
    self.inputSize = inputSize
    self.hiddenSize = hiddenSize
    self.seqLength = 1
    self.miniBatch = 1
    self.numLayers = numLayers
    self.bidirectional = 'CUDNN_BIDIRECTIONAL'
    self.inputMode = 'CUDNN_LINEAR_INPUT'
    self.mode = 'CUDNN_LSTM'
    self.dropout = 0
    self.seed = 0x01234567


    self.gradInput = torch.CudaTensor()
    self.output = torch.CudaTensor()
    self.weight = torch.CudaTensor()
    self.gradWeight = torch.CudaTensor()
    self.reserve = torch.CudaTensor()
    self.hiddenOutput = torch.CudaTensor()
    self.cellOutput = torch.CudaTensor()
    self.gradHiddenInput = torch.CudaTensor()
    self.gradCellInput = torch.CudaTensor()

    self:training()
    self:reset()


end
-- Add * 2 to hidden size since we are implementing a BRNN which concats outputs of both modules.
function BLSTM:resizeOutput(tensor)
    return tensor:resize(self.seqLength, self.miniBatch, self.hiddenSize * 2)
end

function BLSTM:resizeHidden(tensor)
    return tensor:resize(self.numLayers, self.miniBatch, self.hiddenSize * 2)
end

function BLSTM:resetIODescriptors()
    self.xDescs = self:createTensorDescriptors(self.seqLength)
    self.yDescs = self:createTensorDescriptors(self.seqLength)

    for i = 0, self.seqLength - 1 do
        local dim = torch.IntTensor({ self.inputSize, self.miniBatch, self.seqLength })
        local stride = torch.IntTensor({ 1, dim[1], dim[1] * dim[2] })
        errcheck('cudnnSetTensorNdDescriptor',
            self.xDescs[i],
            self.datatype,
            3,
            dim:data(),
            stride:data())
        -- Add * 2 to hidden size since we are implementing a BRNN which concats outputs of both modules.
        local dim = torch.IntTensor({ self.hiddenSize * 2, self.miniBatch, self.seqLength })
        local stride = torch.IntTensor({ 1, dim[1], dim[1] * dim[2] })
        errcheck('cudnnSetTensorNdDescriptor',
            self.yDescs[i],
            self.datatype,
            3,
            dim:data(),
            stride:data())
    end
end