import matplotlib.pyplot as plt
import matplotlib.animation as animation

def denoising_animation(X):
    fig = plt.figure()
    ims = []
    for i in range(len(X)):
        im = plt.imshow(X[i], cmap="magma", animated=True)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    return animate

def noising_animation(X):
    pass

def noising_multiplot(X, sections=5):
    pass

