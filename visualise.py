import matplotlib.pyplot as plt
import matplotlib.animation as animation

def denoising_animation(X):
    fig = plt.figure()
    ims = []
    for i in range(len(X)):
        im = plt.imshow(X[i][0].cpu(), cmap="magma", animated=True, vmin=-4, vmax=4)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    return animate

def noising_animation(X):
    pass

def noising_multiplot(X, sections=5):
    pass

