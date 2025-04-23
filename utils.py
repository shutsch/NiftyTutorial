import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def load_psf(space):
    hsp = space.get_default_codomain()
    psf = lambda k: 1./(1.+(k/20.)**2)/space.size
    PD = ift.PowerDistributor(hsp)
    psf = ift.PS_field(PD.domain[0], psf)
    psf = PD(psf)
    ht = ift.HartleyOperator(hsp, space)
    pos_psf = ht(psf).val
    pos_psf = np.roll(pos_psf, pos_psf.shape[0]//2, axis = 0)
    pos_psf = np.roll(pos_psf, pos_psf.shape[1]//2, axis = 1)
    pos_psf = ift.makeField(space, pos_psf)
    R = ht @ ift.DiagonalOperator(psf) @ ht.adjoint
    return R, pos_psf.val

def plot_2D(inp, label):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (10, 8))
    im = ax.imshow(inp.T, origin='lower', extent=[0,1,0,1])
    ax.set_title(label)
    fig.colorbar(im, ax=ax)
    plt.show()

def geovi_sampling(likelihood):
    ic_samp = ift.AbsDeltaEnergyController(1E-3, iteration_limit = 30)
    ic_sampnl = ift.AbsDeltaEnergyController(0.1, iteration_limit = 20,
                                            convergence_level=2)
    mini_samp = ift.NewtonCG(ic_sampnl)
    ic_mini = ift.AbsDeltaEnergyController(0.1, iteration_limit=15,
                                            name = 'Minimizer')
    minimizer = ift.NewtonCG(ic_mini)

    N_samples = 4
    iteration_limit = 5
    initial_mean = 0.1 * ift.from_random(likelihood.domain)
    samples = ift.optimize_kl(likelihood, iteration_limit, N_samples,
                            minimizer, ic_samp, mini_samp,
                            initial_position=initial_mean)
    ev, _ = ift.estimate_evidence_lower_bound(
    ift.StandardHamiltonian(likelihood), samples, 
    min(100, likelihood.domain.size), verbose = False)
    return samples, ev.average().val[()]


from matplotlib.colors import LogNorm
def plot_posterior(samples, data, model3, diffuse, model2, pspec):
    fig, ax = plt.subplots(nrows=5, ncols=3, figsize = (20, 28))

    ax[0,0].set_visible(False)
    ax[0,2].set_visible(False)
    ax[4,0].set_visible(False)
    ax[4,2].set_visible(False)

    im = ax[0,1].imshow(data['data'].T, origin='lower', extent=[0,1,0,1],
                        norm = LogNorm(vmin=0.6, vmax=20.), cmap='magma')
    ax[0,1].set_title('Data')
    fig.colorbar(im, ax=ax[0,1])

    im = ax[1,0].imshow(data['sky'].T, origin='lower', extent=[0,1,0,1],
                        norm = LogNorm(vmin=3., vmax=100.) , cmap='magma')
    ax[1,0].set_title('Ground truth')
    fig.colorbar(im, ax=ax[1,0])

    im = ax[1,1].imshow(samples.average(model3.force).val.T, origin='lower', extent=[0,1,0,1],
                        norm = LogNorm(vmin=3., vmax=100.) , cmap='magma')
    ax[1,1].set_title('Posterior mean')
    fig.colorbar(im, ax=ax[1,1])
    sm = tuple(s for s in samples.iterator(model3.force))
    im = ax[1,2].imshow(sm[0].val.T, origin='lower', extent=[0,1,0,1],
                        norm = LogNorm(vmin=3., vmax=100.), cmap='magma')
    ax[1,2].set_title('Posterior sample')
    fig.colorbar(im, ax=ax[1,2])


    im = ax[2,0].imshow(data['points'].T, origin='lower', extent=[0,1,0,1],
                        norm = LogNorm(vmin=3., vmax=100.), cmap='magma')
    ax[2,0].set_title('Ground truth (sources)')
    fig.colorbar(im, ax=ax[2,0])

    im = ax[2,1].imshow(samples.average(model2.force).val.T, origin='lower', extent=[0,1,0,1],
                        norm = LogNorm(vmin=3., vmax=100.), cmap='magma')
    ax[2,1].set_title('Posterior mean (sources)')
    fig.colorbar(im, ax=ax[2,1])

    sm = tuple(s for s in samples.iterator(model2.force))
    im = ax[2,2].imshow(sm[0].val.T, origin='lower', extent=[0,1,0,1],
                        norm = LogNorm(vmin=3., vmax=100.), cmap='magma')
    ax[2,2].set_title('Posterior sample (sources)')
    fig.colorbar(im, ax=ax[2,2])


    im = ax[3,0].imshow(data['diffuse'].T, origin='lower', extent=[0,1,0,1],
                        norm = LogNorm(vmin=0.6, vmax=20.), cmap='magma')
    ax[3,0].set_title('Ground truth (diffuse)')
    fig.colorbar(im, ax=ax[3,0])

    im = ax[3,1].imshow(samples.average(diffuse.force).val.T, origin='lower', extent=[0,1,0,1],
                        norm = LogNorm(vmin=0.6, vmax=20.), cmap='magma')
    ax[3,1].set_title('Posterior mean (diffuse)')
    fig.colorbar(im, ax=ax[3,1])

    sm = tuple(s for s in samples.iterator(diffuse.force))
    im = ax[3,2].imshow(sm[0].val.T, origin='lower', extent=[0,1,0,1],
                        norm = LogNorm(vmin=0.6, vmax=20.), cmap='magma')
    ax[3,2].set_title('Posterior sample (diffuse)')
    fig.colorbar(im, ax=ax[3,2])

    ax = ax[4,1]
    ks = pspec.target[0].k_lengths
    lbl = 'posterior samples'
    for i,s in enumerate(samples.iterator()):
        ss = pspec.force(s)
        ax.plot(ks[1:], ss.val[1:], color='k', alpha = 0.4,
                label=lbl)
        lbl=None

    ax.plot(ks[1:],data['pspec'][1:], color = 'g',label='ground truth')
    mm = samples.average((pspec.log()).force)
    ax.plot(ks[1:], mm.exp().val[1:], color='r',label='posterior mean')
    #ax.set_ylim([1E-2*np.min(pp.val[1:]), 10*np.max(pp.val[1:])])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$|k|$')
    ax.set_ylabel(r'$P_s\left(|k|\right)$')
    ax.set_title('Power spectrum')
    leg = ax.legend()
    #for lh in leg.legendHandles: 
    #    lh.set_alpha(1)
    fig.tight_layout()
    plt.show()
