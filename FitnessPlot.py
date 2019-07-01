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
            #checkpoint.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            checkpoint.add_reporter(stats)
            print(checkpoint.generation_statistics())
        return result


plot = FitnessPlot()
plot._create_plot_list()
