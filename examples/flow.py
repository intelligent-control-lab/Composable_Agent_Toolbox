import random
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import sklearn.cluster
import statistics

if __name__ == '__main__':

    v0_left = []
    v0_mid = []
    v0_right = []
    v0_avg = []

    vf_left = []
    vf_mid = []
    vf_right = []
    vf_avg = []

    dv0 = []
    dvf = []

    for iter in range(100):

        print(iter)

        vel = []
        a = random.randint(1, 3)
        b = random.randint(1, 3)
        c = random.randint(1, 3)
        for i in range(a + b + c):
            vel.append(random.randint(25, 39) + random.random())

        v0_left.append(min(vel[:a]))
        v0_mid.append(min(vel[a:a+b]))
        v0_right.append(min(vel[a+b:]))
        v0_avg.append((v0_left[-1] + v0_mid[-1] + v0_right[-1]) / 3) 
        # actually, overall velocity of a lane isn't just pick the front velocity.
        # we need to make a random arrangement of the cars and then the velocity of each is
        # min(desired velocity, velocity of front car). then velocity of lane is
        # velocity of slowest car in lane OR average velocity of all cars in lane

        dv0.append(statistics.mean(
                    [v - v0_left[-1] for v in vel[:a]] 
                   + [v - v0_mid[-1] for v in vel[a:a+b]] 
                   + [v - v0_right[-1] for v in vel[a+b:]]))

        kmeans = sklearn.cluster.KMeans(n_clusters=3).fit(np.asarray(vel).reshape(-1, 1))
        print(vel)
        print(kmeans.labels_)
        opt_vel_left = [vel[i] for i in range(a + b + c) if kmeans.labels_[i] == 0]
        opt_vel_mid = [vel[i] for i in range(a + b + c) if kmeans.labels_[i] == 1]
        opt_vel_right = [vel[i] for i in range(a + b + c) if kmeans.labels_[i] == 2]
        print(opt_vel_left, opt_vel_mid, opt_vel_right)
        vf_left.append(min(opt_vel_left) * 0.97)
        vf_mid.append(min(opt_vel_mid) * 0.97)
        vf_right.append(min(opt_vel_right) * 0.97)
        vf_avg.append((vf_left[-1] + vf_mid[-1] + vf_right[-1]) / 3)
        print("INIT: ", v0_left, v0_mid, v0_right, v0_avg)
        print("FINAL: ", vf_left, vf_mid, vf_right, vf_avg)

        dvf.append(statistics.mean(
                    [v - vf_left[-1] for v in opt_vel_left] 
                   + [v - vf_mid[-1] for v in opt_vel_mid] 
                   + [v - vf_right[-1] for v in opt_vel_right]))
        
        print(dv0, dvf)

    ttest1 = scipy.stats.ttest_rel(v0_avg, vf_avg)
    print(ttest1)

    ttest2 = scipy.stats.ttest_rel(dv0, dvf)
    print(ttest2)

    # set up the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(0,5)
    ax.set_ylim(0,10)

    # draw lines
    xmin = 0
    xmax = 5
    y = 5
    height = 1

    plt.hlines(y, xmin, xmax)
    plt.vlines(xmin, y - height / 2., y + height / 2.)
    plt.vlines(xmax, y - height / 2., y + height / 2.)

    plt.scatter(dv0, [y for _ in range(len(dv0))], label='initial')
    plt.scatter(dvf, [y for _ in range(len(dv0))], label='final')

    # add numbers
    plt.text(xmin - 0.1, y, '0', horizontalalignment='right')
    plt.text(xmax + 0.1, y, '5', horizontalalignment='left')

    ax.legend(loc='lower left')

    plt.axis('off')
    plt.show()
