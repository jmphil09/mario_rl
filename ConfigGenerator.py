class ConfigGenerator:
    def __init__(
        self,
        filename='config_file',
        fitness_criterion='max',
        fitness_threshold=100000,
        pop_size=10,
        reset_on_extinction=False
    ):
        with open(filename, 'w') as fn:
            file_data = """
                [NEAT]
                fitness_criterion     = {}
                fitness_threshold     = {}
                pop_size              = {}
                reset_on_extinction   = {}
            """.format(
                    fitness_criterion,
                    fitness_threshold,
                    pop_size,
                    reset_on_extinction
                )
            file_data = '\n'.join(line.lstrip() for line in file_data.splitlines()[1::])
            fn.write(file_data)


test = ConfigGenerator()
