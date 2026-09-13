"""
CKNNA representation similarity analysis for REPA-Seg.

Inspired by on Yu et al., 2025:

Original source implementation: 
    https://github.com/sihyun-yu/REPA 
Paper: 
    S. Yu et al., "Representation Alignment for Generation: 
    Training Diffusion Transformers Is Easier Than You Think," ICLR, 2025. 
    (https://proceedings.iclr.cc/paper_files/paper/2025/file/d9e42b4d7163931f3689d6d6fbaa11d0-Paper-Conference.pdf)
- see github issue, for computational question: https://github.com/sihyun-yu/REPA/issues/24

CKNNA implementation based on Huh et al., 2024
Source code: 
    https://github.com/minyoungg/platonic-rep/blob/main/metrics.py
Paper:  
    M. Huh et al., "The platonic representation hypothesis", PMLR, 2024. 
    (https://proceedings.mlr.press/v235/huh24a.html)

Computation of the unbiased Hilbert-Schmidt Independence Criterion (HSIC)
    Reference: https://github.com/minyoungg/platonic-rep/blob/main/metrics.py
    Primary-Reference: Song et al., "Feature Selection via Dependence Maximization", 2012.
                       https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
"""
import torch

# CKNNA COMPUTATION
def similarity(Ks, Lt, k_neighbors, device):
    m = Ks.shape[0]                                  
    Ks_hat = Ks.clone().fill_diagonal_(float("-inf"))
    Lt_hat = Lt.clone().fill_diagonal_(float("-inf"))

    # get k nearest neighbors indices for each row
    _, k_Ks_indices = torch.topk(Ks_hat, k_neighbors, dim=1)
    _, k_Lt_indices = torch.topk(Lt_hat, k_neighbors, dim=1)
    
    # create masks for nearest neighbors
    mask_Ks = torch.zeros(m, m, device=device).scatter_(1, k_Ks_indices, 1)
    mask_Lt = torch.zeros(m, m, device=device).scatter_(1, k_Lt_indices, 1)
    
    # intersection of nearest neighbors
    mask = mask_Ks * mask_Lt
                
    sim = hsic_unbiased(mask * Ks, mask * Lt)
    return sim

@staticmethod
def cknna_formula(Ks, Lt, k_neighbors, device):
    """ 
    CKNNA similarity.
    - Input: Ks is matrix of image vectors for the student, Lt for the teacher
        - e.g. Ks with Ks.shape[0] = m images with row length shape[1] = n of hidden dimension D_s
    """
    if Ks.shape[0] != Lt.shape[0]:
        raise ValueError("Ks and Lt must contain the same number of image observations")

    (Ks, Lt) = (Ks.float(), Lt.float())

    # Pairwise similarity matrix among all observations in the matrix (self-similarity)
    Ks = Ks @ Ks.T
    Lt = Lt @ Lt.T
    
    sim_kl = similarity(Ks, Lt, k_neighbors, device)
    sim_kk = similarity(Ks, Ks, k_neighbors, device)
    sim_ll = similarity(Lt, Lt, k_neighbors, device)
            
    return sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()


def hsic_unbiased(Ks, Lt):
    """
    Compute the unbiased Hilbert-Schmidt Independence Criterion (HSIC)
    """
    m = Ks.shape[0]

    Ks_tilde = Ks.clone().fill_diagonal_(0)
    Lt_tilde = Lt.clone().fill_diagonal_(0)

    # Compute HSIC
    HSIC_value = (
        (torch.sum(Ks_tilde * Lt_tilde.T))
        + (torch.sum(Ks_tilde) * torch.sum(Lt_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(Ks_tilde, Lt_tilde)) / (m - 2))
    )

    return HSIC_value / (m * (m-3))
