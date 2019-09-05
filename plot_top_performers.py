from FitnessPlot import FitnessPlot


'''
for n in range(1,6):
    plot = FitnessPlot(folder_prefix='data_top{}'.format(n))
    plot.plot_all_workers()
    plot.plot_workers_as_average()
'''

plot = FitnessPlot(folder_prefix='data_top1_100gen', num_workers=32)
worker_dict = plot.create_worker_dict()
#plot.plot_all_workers()
#plot.plot_workers_as_average()
#print(worker_dict)
for key,value in worker_dict.items():
    dict_len = len(value)
    #if dict_len < 100:
    #    print(key)
    #    print(dict_len)
    print(key)
    print(value[len(value)-1])
