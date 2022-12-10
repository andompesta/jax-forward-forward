from matplotlib import pyplot as plt


def show_image(img):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    ax.imshow(img)
    fig.show()
