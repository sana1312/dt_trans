""" Implementation of all available options """
from __future__ import print_function

## Training parameters
def train_opts(parser):
    
    # Common training options
    group = parser.add_argument_group('Training_options')
    group.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    group.add_argument('--num-epoch', type=int, default=200,
                       help='Number of training steps')
    group.add_argument('--starting-epoch', type=int, default=1,
                       help="Training from given starting epoch")
    group.add_argument('--cuda-device', type=int, default=0,
                       help='Specify the CUDA device to use')
    group.add_argument('--validation-interval', type=int, default=1,
                          help='Run validation every n epochs')
    
    # Input output settings
    group = parser.add_argument_group('Input-Output')
    group.add_argument('--data-path', required=True,
                       help="""Input data path""")
    group.add_argument('--save-directory', default='train',
                       help="""Result save directory""")

    # Model settings
    group = parser.add_argument_group('Model')
    group.add_argument('-N', type=int, default=6,
                       help="number of encoder and decoder")
    group.add_argument('-H', type=int, default=8,
                       help="heads of attention")
    group.add_argument('-d-model', type=int, default=256,
                       help="embedding dimension, model dimension")
    group.add_argument('-d-ff', type=int, default=2048,
                       help="dimension in feed forward network")
    # Regularization
    group.add_argument('--dropout', type=float, default=0.1,
                       help="Dropout probability; applied in LSTM stacks.")
    group.add_argument('--label-smoothing', type=float, default=0.0,
                       help="""Label smoothing value epsilon.
                       Probabilities of all non-true labels
                       will be smoothed by epsilon / (vocab_size - 1).
                       Set to zero to turn off label smoothing.
                       For more detailed information, see:
                       https://arxiv.org/abs/1512.00567""")
    # Optimization options
    group = parser.add_argument_group('Optimization')
    group.add_argument('--factor', type=float, default=1.0,
                       help="""Factor multiplied to the learning rate scheduler formula in NoamOpt. 
                       For more information about the formula, 
                       see paper Attention Is All You Need https://arxiv.org/pdf/1706.03762.pdf""")
    group.add_argument('--warmup-steps', type=int, default=4000,
                       help="""Number of warmup steps for custom decay.""")
    group.add_argument('--adam-beta1', type=float, default=0.9,
                       help="""The beta1 parameter for Adam optimizer""")
    group.add_argument('--adam-beta2', type=float, default=0.98,
                       help="""The beta2 parameter for Adam optimizer""")
    group.add_argument('--adam-eps', type=float, default=1e-9,
                       help="""The eps parameter for Adam optimizer""")


 
## Evaluation parameters
def generate_opts(parser):
    """Input output settings"""
    group = parser.add_argument_group('Input-Output')
    group.add_argument('--data-path', required=True,
                       help="""Input data path""")
    group.add_argument('--test-file-name', required=True, help="""test file name without .csv""")
    group.add_argument('--save-directory', default='evaluation',
                       help="""Result save directory""")
    group.add_argument('--cuda-device', type=int, default=0,
                       help='Specify the CUDA device to use')
    # Model to be used for generating molecules
    group = parser.add_argument_group('Model')
    group.add_argument('--model-path', help="""Model path""", required=True)
    group.add_argument('--epoch', type=int, help="""Which epoch to use""", required=True)
    # General
    group = parser.add_argument_group('General')
    group.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    group.add_argument('--num-samples', type=int, default=50,
                       help='Number of molecules to be generated')
    group.add_argument('--decode-type',type=str, default='multinomial',help='decode strategy')


