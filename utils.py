import matplotlib.pyplot as plt
from torchvision.utils import save_image
from IPython.core.display import Image, display

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))
    y1 = hist['D_losses']
    y2 = hist['G_losses']
    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend() #loc=1
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()        
        
def make_images(G, z, x, c):
    test_images = G(z, x)
    test_images = (test_images.cpu() + 1) * 0.5
    test_images = test_images.data.view(test_images.shape[0], c, 32, 32)
    return test_images

def show_images(images, path='result.png', rows=8):
    save_image(images, path, nrow=rows, padding=1, pad_value=1.0, scale_each=False, normalize=False)
    display(Image(path))
    
    
    