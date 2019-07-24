from pathlib import Path


class ConfigGenerator:
    def __init__(
        self,
        filename='config',
        fitness_criterion='max',
        fitness_threshold=100000,
        pop_size=20,
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
        bias_replace_rate=0.1,
        compatibility_disjoint_coefficient=1.0,
        compatibility_weight_coefficient=0.5,
        conn_add_prob=0.5,
        conn_delete_prob=0.5,
        enabled_default=True,
        enabled_mutate_rate=0.01,
        feed_forward=True,
        initial_connection="full",
        node_add_prob=0.2,
        node_delete_prob=0.2,
        num_hidden=0,
        num_inputs=840,
        num_outputs=9,
        response_init_mean=1.0,
        response_init_stdev=0.0,
        response_max_value=30.0,
        response_min_value=-30.0,
        response_mutate_power=0.0,
        response_mutate_rate=0.0,
        response_replace_rate=0.0,
        weight_init_mean=0.0,
        weight_init_stdev=1.0,
        weight_max_value=30,
        weight_min_value=-30,
        weight_mutate_power=0.5,
        weight_mutate_rate=0.8,
        weight_replace_rate=0.1,
        compatibility_threshold=3.0,
        species_fitness_func="max",
        max_stagnation=20,
        species_elitism=2,
        elitism=2,
        survival_threshold=0.2
    ):
        self.filename = Path(filename)
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
        self.compatibility_disjoint_coefficient = compatibility_disjoint_coefficient
        self.compatibility_weight_coefficient = compatibility_weight_coefficient
        self.conn_add_prob = conn_add_prob
        self.conn_delete_prob = conn_delete_prob
        self.enabled_default = enabled_default
        self.enabled_mutate_rate = enabled_mutate_rate
        self.feed_forward = feed_forward
        self.initial_connection = initial_connection
        self.node_add_prob = node_add_prob
        self.node_delete_prob = node_delete_prob
        self.num_hidden = num_hidden
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.response_init_mean = response_init_mean
        self.response_init_stdev = response_init_stdev
        self.response_max_value = response_max_value
        self.response_min_value = response_min_value
        self.response_mutate_power = response_mutate_power
        self.response_mutate_rate = response_mutate_rate
        self.response_replace_rate = response_replace_rate
        self.weight_init_mean = weight_init_mean
        self.weight_init_stdev = weight_init_stdev
        self.weight_max_value = weight_max_value
        self.weight_min_value = weight_min_value
        self.weight_mutate_power = weight_mutate_power
        self.weight_mutate_rate = weight_mutate_rate
        self.weight_replace_rate = weight_replace_rate
        self.compatibility_threshold = compatibility_threshold
        self.species_fitness_func = species_fitness_func
        self.max_stagnation = max_stagnation
        self.species_elitism = species_elitism
        self.elitism = elitism
        self.survival_threshold = survival_threshold

    def write_file(self):
        with open(self.filename, 'w') as fn:
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

                # genome compatibility options
                compatibility_disjoint_coefficient = {}
                compatibility_weight_coefficient   = {}

                # connection add/remove rates
                conn_add_prob           = {}
                conn_delete_prob        = {}

                # connection enable options
                enabled_default         = {}
                enabled_mutate_rate     = {}

                feed_forward            = {}
                initial_connection      = {}

                # node add/remove rates
                node_add_prob           = {}
                node_delete_prob        = {}

                # network parameters
                num_hidden              = {}
                num_inputs              = {}
                num_outputs             = {}

                # node response options
                response_init_mean      = {}
                response_init_stdev     = {}
                response_max_value      = {}
                response_min_value      = {}
                response_mutate_power   = {}
                response_mutate_rate    = {}
                response_replace_rate   = {}

                # connection weight options
                weight_init_mean        = {}
                weight_init_stdev       = {}
                weight_max_value        = {}
                weight_min_value        = {}
                weight_mutate_power     = {}
                weight_mutate_rate      = {}
                weight_replace_rate     = {}

                [DefaultSpeciesSet]
                compatibility_threshold = {}

                [DefaultStagnation]
                species_fitness_func = {}
                max_stagnation       = {}
                species_elitism      = {}

                [DefaultReproduction]
                elitism            = {}
                survival_threshold = {}

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
                    self.bias_replace_rate,
                    self.compatibility_disjoint_coefficient,
                    self.compatibility_weight_coefficient,
                    self.conn_add_prob,
                    self.conn_delete_prob,
                    self.enabled_default,
                    self.enabled_mutate_rate,
                    self.feed_forward,
                    self.initial_connection,
                    self.node_add_prob,
                    self.node_delete_prob,
                    self.num_hidden,
                    self.num_inputs,
                    self.num_outputs,
                    self.response_init_mean,
                    self.response_init_stdev,
                    self.response_max_value,
                    self.response_min_value,
                    self.response_mutate_power,
                    self.response_mutate_rate,
                    self.response_replace_rate,
                    self.weight_init_mean,
                    self.weight_init_stdev,
                    self.weight_max_value,
                    self.weight_min_value,
                    self.weight_mutate_power,
                    self.weight_mutate_rate,
                    self.weight_replace_rate,
                    self.compatibility_threshold,
                    self.species_fitness_func,
                    self.max_stagnation,
                    self.species_elitism,
                    self.elitism,
                    self.survival_threshold
                )
            file_data = '\n'.join(line.lstrip() for line in file_data.splitlines()[1::])
            fn.write(file_data)

    def write_all_configs(self, num_workers=16):
        self.write_file()
        with open(self.filename, 'r') as orig_config:
            config_data = orig_config.read()
            for config_file_name in [Path(str(self.filename) + '_' + str(worker_num)) for worker_num in range(num_workers)]:
                with open(config_file_name, 'w') as fn:
                    fn.write(config_data)
