local LSTM, parent = torch.class('cudnn.LSTM', 'cudnn.RNN')

function LSTM:__init(inputSize, hiddenSize, numLayers)
    parent.__init(self,inputSize, hiddenSize, numLayers)
    self.mode = 'CUDNN_LSTM'
    self:reset()
end
