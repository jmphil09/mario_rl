class ConfigGenerator:
    def __init__(
        self,
        filename='config_file',
        fitness_criterion='max',
        fitness_threshold=100000,
        pop_size=10,
        reset_on_extinction=False,
        activation_default="sigmoid",
        activation_mutate_rate=0.0,
        activation_options="sigmoid",
        aggregation_default="sum",
        aggregation_mutate_rate=0.0,
        aggregation_options="sum",
        bias_init_mean=0.0,
        bias_init_stdev=1.0,
        bias_max_value=30.0,
        bias_min_value=-30.0,
        bias_mutate_power=0.5,
        bias_mutate_rate=0.7,
        bias_replace_rate=0.1
    ):
        self.filename = filename
        self.fitness_criterion = fitness_criterion
        self.fitness_threshold = fitness_threshold
        self.pop_size = pop_size
        self.reset_on_extinction = reset_on_extinction
        self.activation_default = activation_default
        self.activation_mutate_rate = activation_mutate_rate
        self.activation_options = activation_options
        self.aggregation_default = aggregation_default
        self.aggregation_mutate_rate = aggregation_mutate_rate
        self.aggregation_options = aggregation_options
        self.bias_init_mean = bias_init_mean
        self.bias_init_stdev = bias_init_stdev
        self.bias_max_value = bias_max_value
        self.bias_min_value = bias_min_value
        self.bias_mutate_power = bias_mutate_power
        self.bias_mutate_rate = bias_mutate_rate
        self.bias_replace_rate = bias_replace_rate

        with open(filename, 'w') as fn:
            file_data = """
                [NEAT]
                fitness_criterion     = {}
                fitness_threshold     = {}
                pop_size              = {}
                reset_on_extinction   = {}

                [DefaultGenome]
                # node activation options
                activation_default      = {}
                activation_mutate_rate  = {}
                activation_options      = {}

                # node aggregation options
                aggregation_default     = {}
                aggregation_mutate_rate = {}
                aggregation_options     = {}

                # node bias options
                bias_init_mean          = {}
                bias_init_stdev         = {}
                bias_max_value          = {}
                bias_min_value          = {}
                bias_mutate_power       = {}
                bias_mutate_rate        = {}
                bias_replace_rate       = {}
            """.format(
                    self.fitness_criterion,
                    self.fitness_threshold,
                    self.pop_size,
                    self.reset_on_extinction,
                    self.activation_default,
                    self.activation_mutate_rate,
                    self.activation_options,
                    self.aggregation_default,
                    self.aggregation_mutate_rate,
                    self.aggregation_options,
                    self.bias_init_mean,
                    self.bias_init_stdev,
                    self.bias_max_value,
                    self.bias_min_value,
                    self.bias_mutate_power,
                    self.bias_mutate_rate,
                    self.bias_replace_rate
                )
            file_data = '\n'.join(line.lstrip() for line in file_data.splitlines()[1::])
            fn.write(file_data)


test = ConfigGenerator()
