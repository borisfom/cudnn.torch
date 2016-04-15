local BLSTM, parent = torch.class('cudnn.LSTM', 'cudnn.RNN')

function BLSTM:__init(inputSize, hiddenSize, numLayers)
    parent.__init(self,inputSize, hiddenSize, numLayers)
    self.mode = 'CUDNN_LSTM'
    self:reset()
end
