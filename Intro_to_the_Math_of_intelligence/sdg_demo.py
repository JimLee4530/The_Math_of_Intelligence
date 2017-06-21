import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_data(filename):
    df = pd.read_csv(filename)
    return df.values[:,[2,7]]

def compute_error(point, b, m):
    X = point[:, 1]
    y = point[:, 0]
    error = sum(((X*m + b) - y)**2)/2
    return error

def step_gradient(point, m, b, learning_rate, batch_size):
    dm = 0
    db = 0
    X = point[:, 1]
    y = point[:, 0]
    h = X*m + b
    dm = sum(X * (h- y))/batch_size
    db = sum(h - y)/batch_size
    m += - learning_rate * dm
    b += - learning_rate * db
    return m, b

def run(point, m, b, learning_rate, batch_size, num_itertion):
    err_his = np.zeros(num_itertion)
    for i in range(num_itertion):
        point_rand = np.random.choice(range(batch_size), batch_size)
        point_batch = point[point_rand]
        m, b = step_gradient(point_batch, m, b, learning_rate, batch_size)
        err_his[i] = compute_error(point, b, m)
    return m, b, err_his

if __name__ == '__main__':
    filename = '2017.csv'
    point = get_data(filename)
    m = 0
    b = 0
    learning_rate = 0.0001
    batch_size = 30
    num_itertion = 10000

    # plt.figure()
    # plt.scatter(point[:, 1], point[:, 0])
    # plt.title('point')
    # plt.show()

    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(b, m,compute_error(point, b, m))
    print "Running..."
    m, b, his = run(point, m, b, learning_rate, batch_size, num_itertion)
    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_itertion, b, m,compute_error(point, b, m))

    x = np.linspace(0, 1.0, 100)
    y = m * x + b
    plt.figure()
    plt.subplot(211)
    plt.scatter(point[:, 1], point[:, 0])
    plt.plot(x, y, 'b', lw=1)
    plt.title('line')
    plt.subplot(212)
    x = np.linspace(0, 10000, 10000)
    plt.plot(x, his)
    plt.title('error')
    plt.show()