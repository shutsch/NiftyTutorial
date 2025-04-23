import matplotlib.pyplot as plt
import numpy as np


def plot(fields, spectra,  distances, title=None):

    n_samples = len(fields)
    fig = plt.figure()#tight_layout=True)
    if title is not None:
        fig.suptitle(title, fontsize=14)
    
    fig, ax = plt.subplots(nrows=n_samples, ncols=2, figsize=(10, n_samples*3))
    # Field
    
    for j, field in enumerate(fields):
        shp = field.shape
        ax1 = ax[j, 0] if len(ax.shape) == 2 else ax[0]
        # ax1.axhline(y=0., color='k', linestyle='--', alpha=0.25) 
        im = ax1.imshow(field, origin="lower")
        ax1.set_xticks(np.linspace(0, shp[0], 5), np.linspace(0, shp[0], 5) * distances[0])
        ax1.set_yticks(np.linspace(0, shp[1], 5), np.linspace(0, shp[1], 5) * distances[1])
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Field realizations')
        fig.colorbar(im, ax=ax1)
    
    # Spectrum
        ax2 = ax[j, 1] if len(ax.shape) == 2 else ax[1]
        spectrum = spectra[j]
        xcoord = np.arange(len(spectrum))*1/distances[0] 
        spectrum = spectrum.at[0].set(spectrum[1])
        ax2.plot(xcoord, spectrum)
        ax2.set_ylim(1e-2, 1e6)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('k')
        ax2.set_ylabel('p(k)')
        ax2.set_title('Power Spectrum')
        
    fig.align_labels()
    plt.show()