local BLSTM, parent = torch.class('cudnn.BLSTM', 'cudnn.RNN')

function BLSTM:__init(inputSize, hiddenSize, numLayers)
    parent.__init(self,inputSize, hiddenSize, numLayers)
    self.bidirectional = 'CUDNN_BIDIRECTIONAL'
    self.mode = 'CUDNN_LSTM'
    self.numDirections = 2
    self:reset()
end
