def plot_activations (x): #plots activation results of CNN layers
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)
    block_pool_features = model.predict(x)
    print np.shape(block_pool_features)
    l = 32
    x, y, k = 4, 8, 1  # x=l1 for showing weights of l1 one in one row
    if x*y != l:
        x = x+1
    fig = plt.figure()
    for i in range(l):
        if x > y:
            ax = fig.add_subplot(y, x, k)
        else:
            ax = fig.add_subplot(x, y, k)
        ax.matshow(block_pool_features[0, :, :, k], cmap=matplotlib.cm.binary)
        plt.xticks(np.asarray([]))
        plt.yticks(np.asarray([]))
        k = k + 1
    return plt

plt.show(plot_activations(x))
