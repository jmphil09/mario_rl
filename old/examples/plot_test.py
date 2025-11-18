from FitnessPlot import FitnessPlot


plot = FitnessPlot(folder_prefix='hyperparam_data/1565232031', num_workers=8)
worker_dict = plot.create_worker_dict()

plot.plot_all_workers()
plot.plot_workers_as_average()
