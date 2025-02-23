"""
Script containing tools for image viewing in Jupyter Notebooks, mainly with Matplotlib's PyPlot
"""
import matplotlib.pyplot as plt

# Put these in another util files later!
def pyplot_fig(img, cmap="hot", title=None, extra_scatter=None, figsize=(11,5)):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap)
    
    if title is not None:
        plt.title(title)
    
    plt.show()
    
def pyplot_img_w_scatter(img, scatter, cmap="hot", scatter_color='b', title=None, figsize=(11,5)):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap)
    
    if title is not None:
        plt.title(title)
    
    #note: numpy and pyplot's xy coordinate is inverted
    plt.scatter(x=scatter[1], y=scatter[0], c=scatter_color, s=0.5)
    plt.show()