"""
========
Barchart
========

A bar plot with errorbars and height labels on individual bars
"""
import matplotlib.pyplot as plt
import numpy as np

N = 10
ind = np.arange(N)  # the x locations for the groups
width = 0.2  # the width of the bars

fig, ax = plt.subplots()

# Homo
# - Method : homo l1
# - Whole Home
# avg_acc :0.1567 - 0.0005
# recall :0.0045 - 0.0005
# f1 :0.0088 - 0.0010
# nde :0.8433 - 0.0005
#
# - Method : homo l2
# - Whole Home
# avg_acc :0.1595 - 0.0047
# recall :0.0047 - 0.0038
# f1 :0.0090 - 0.0067
# nde :0.8405 - 0.0047
#
#
# - Method : homo dense
# - Whole Home
# avg_acc :0.1759 - 0.0307
# recall :0.0223 - 0.0312
# f1 :0.0350 - 0.0456
# nde :0.8241 - 0.0307

# Heterogenous
# - Method : heter l1
# - Whole Home
# avg_acc :0.1870 - 0.0298
# recall :0.0330 - 0.0298
# f1 :0.0522 - 0.0443
# nde :0.8130 - 0.0298
#
# - Method : heter l2
# - Whole Home
# avg_acc :0.2310 - 0.0867
# recall :0.0766 - 0.0870
# f1 :0.1055 - 0.1042
# nde :0.7690 - 0.0867
#
# - Method : heter dense
# - Whole Home
# avg_acc :0.1999 - 0.1137
# recall :0.0717 - 0.0771
# f1 :0.0990 - 0.0956
# nde :0.8001 - 0.1137




# HOMO
# f1_means = (0.6, 0.6, 0.5, 0.45, 0.33, 0.32, 0.4, 0.0088, 0.0090, 0.0350)
# f1_std = (0.05, 0.05, 0.06, 0.075, 0.04, 0.05, 0.06, 0.0010, 0.0067, 0.0256)
# rects1 = ax.bar(ind, f1_means, width, color='gold', yerr=f1_std)
#
# acc_means = (0.87, 0.9, 0.78, 0.72, 0.7, 0.65, 0.7, 0.1567, 0.1595, 0.1759)
# acc_std = (0.025, 0.02, 0.025, 0.06, 0.07, 0.02, 0.06, 0.0005, 0.0047, 0.0307)
# rects2 = ax.bar(ind + width, acc_means, width, color='violet', yerr=acc_std)
#
# nde_means = (0.61, 0.61, 0.7, 0.75, 0.8, 0.8, 0.76, 0.8433, 0.8405, 0.8241)
# nde_std = (0.1, 0.1, 0.12, 0.15, 0.08, 0.1, 0.12, 0.0005, 0.0047, 0.0307)
# rects3 = ax.bar(ind + 2 * width, nde_means, width, color='slateblue', yerr=nde_std)

# HETER

f1_means = (0.47, 0.49, 0.35, 0.33, 0.23, 0.23, 0.5, 0.0522, 0.1055, 0.0990)
f1_std = (0.025, 0.04, 0.05, 0.08, 0.05, 0.06, 0.06, 0.0443, 0.1042, 0.0956)
rects1 = ax.bar(ind, f1_means, width, color='gold', yerr=f1_std)

acc_means = (0.75, 0.79, 0.65, 0.62, 0.61, 0.56, 0.58, 0.1870, 0.2310, 0.1999)
acc_std = (0.06, 0.03, 0.07, 0.02, 0.07, 0.04, 0.05, 0.0298, 0.0867, 0.1137)
rects2 = ax.bar(ind + width, acc_means, width, color='violet', yerr=acc_std)

nde_means = (0.72, 0.73, 0.86, 0.87, 0.9, 0.89, 0.82, 0.8130, 0.7690, 0.8001)
nde_std = (0.06, 0.06, 0.1, 0.025, 0.09, 0.07, 0.07, 0.0298, 0.0867, 0.1137)
rects3 = ax.bar(ind + 2 * width, nde_means, width, color='slateblue', yerr=nde_std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Avg. F-measure, Accuracy & NDE')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(
    ('VM-CGS', 'VM-GS', 'BSC-MoG', 'DDSC+SF', 'DDSC+TCP+GL', 'DDSC', 'FHMM', 'FC-l1', 'FC-l2', 'FC-dense'), rotation=40)

ax.legend((rects1[0],
           rects2[0],
           rects3[0]),
          ('Avg. F-measure',
           'Accuracy',
           'NDE'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        # ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
        #         '%d' % int(height),
        #         ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.show()
