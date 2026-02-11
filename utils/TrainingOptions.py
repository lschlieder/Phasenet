class TrainingOptions:
    def __init__(self, batch_num = 10, epochs = 10000, scale = 2, N = 60, L = 50, padding = None, transducer_radius = 25, Training = Training, Path = 'utils/', z = '30,30,30' ):
        self.batch_num = batch_num
        self.epochs = epochs
        #self.shuffle_num = 1
        self.scale = scale
        self.N = N
        self.L = L
        self.padding = padding
        self.transducer_radius = transducer_radius  # to avoid confusion should be L/2
        self.Training = Training
        self.PATH = Path
        self.z = '30,30,30'
        first_dist = 30
        last_dist = 30
        train_on_mse = True
        use_FT = True
        use_evanescence_layers = False
        use_fourier_lens = False
        input = 'phase'
        use_image_as_direct_input = False

        use_tanh_activation = False
        jitter = False
        activation = 'none'  # choose from 'relu' 'abs' 'tanh' 'none'/None 'log'
        f0 = 1.0e6
        cM = 1484
        dataset = 'mnist'

        argv = sys.argv[1:]

        learning_rate = 0.01

        optim_str = 'admm'

        save = False

        # loss_fn = mse_cropped
        loss_function_string = 'CCMSE'
        metrics = []
        output_function = output_images