from FitnessPlot import FitnessPlot


plot = FitnessPlot(folder_prefix='data')
worker_dict = plot.create_worker_dict()

plot.plot_all_workers()
plot.plot_workers_as_average()
