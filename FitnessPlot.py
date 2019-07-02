import glob
import neat

from pathlib import Path


class FitnessPlot:
    def __init__(self, config_file_name='config'):
        self.config_file_name = config_file_name

    def _get_worker_checkpoints(self, dir, worker):
        result = None
        filename_str = str(Path('data/{}_{}/worker-{}-neat-checkpoint-*'.format(self.config_file_name, worker, worker)))
        file_list = glob.glob(filename_str)
        result = file_list
        print(result)
        return result

    def _create_plot_list(self):
        result = {}
        checkpoint_files = self._get_worker_checkpoints('',0)
        for file in checkpoint_files:
            checkpoint = neat.Checkpointer.restore_checkpoint(file)
            print(checkpoint)
            #object_methods = [method_name for method_name in dir(checkpoint) if callable(getattr(checkpoint, method_name))]
            #print(object_methods)
            print(dir(checkpoint))
            print(checkpoint.best_genome)
            #checkpoint.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            checkpoint.add_reporter(stats)
            print(checkpoint.generation)
            rep_set = neat.reporting.ReporterSet()
            rep_set.add(reporter=neat.StatisticsReporter())
            print(stats.best_genome())
            print(rep_set.post_evaluate(
                config=checkpoint.config,
                population=checkpoint.population,
                species=checkpoint.species,
                best_genome=checkpoint.best_genome
                )
            )
        return result


plot = FitnessPlot()
plot._create_plot_list()
