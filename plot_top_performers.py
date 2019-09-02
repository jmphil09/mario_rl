from FitnessPlot import FitnessPlot


for n in range(1,6):
    plot = FitnessPlot(folder_prefix='data_top{}'.format(n))
    plot.plot_all_workers()
    plot.plot_workers_as_average()
