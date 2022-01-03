from FitnessPlot_v6 import FitnessPlot


NUM_THREADS = 48

for n in range(1, 9):
    plot = FitnessPlot(num_threads=NUM_THREADS, folder_prefix='v6_level'+ str({}).format(n) + '-1', plot_max_score=True, max_score=3074)
    plot.clean_up_data()

    del plot
