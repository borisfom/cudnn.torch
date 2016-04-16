local RNNReLU, parent = torch.class('cudnn.RNNReLU', 'cudnn.RNN')

function RNNReLU:__init(inputSize, hiddenSize, numLayers)
    parent.__init(self,inputSize, hiddenSize, numLayers)
    self.mode = 'CUDNN_RNN_RELU'
end
