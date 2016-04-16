local RNNTanh, parent = torch.class('cudnn.RNNTanh', 'cudnn.RNN')

function RNNTanh:__init(inputSize, hiddenSize, numLayers)
    parent.__init(self,inputSize, hiddenSize, numLayers)
    self.mode = 'CUDNN_RNN_TANH'
end
