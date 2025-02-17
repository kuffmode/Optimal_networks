import generative as gen
import numpy as np
import scipy
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from utils import density_distance

def check_density(adj):
    return np.sum(adj)/(adj.shape[0])**2


res_parcellation = 1
consensus_mat = scipy.io.loadmat(
    "datasets/human/Consensus_Connectomes.mat",
    simplify_cells=True,
    squeeze_me=True,
    chars_as_strings=True,
)
connectivity = consensus_mat["LauConsensus"]["Matrices"][res_parcellation][0].astype(bool)
coordinates = consensus_mat["LauConsensus"]["Matrices"][res_parcellation][3]
euclidean_dis = squareform(pdist(coordinates, metric='euclidean'))

n_iterations = 10_000
n_nodes = connectivity.shape[0]
noise = np.zeros(n_iterations)
penalty = np.zeros(n_iterations)
batch_size = np.full(n_iterations, 32)

params = np.linspace(1, 21, 21)
scores = np.zeros(len(params))
histories = np.zeros((n_nodes, n_nodes, n_iterations, len(params)))
densities = np.zeros(len(params))
for ind, val in tqdm(enumerate(params), 
                     total=len(params), 
                     desc="blip bloop"):
    print(f"Running for alpha = {val}")
    alpha = np.full(n_iterations, val)
    histories[...,ind] = gen.simulate_network_evolution(
        coordinates=coordinates,
        n_iterations=n_iterations,
        alpha=alpha,
        beta=np.full(n_iterations, 1),
        noise=noise,
        distance_fn=gen.resistance_distance,
        connectivity_penalty=penalty,
        n_jobs=-1,
        random_seed=11,
        batch_size=batch_size,
    )
    scores[ind] = density_distance(connectivity, histories[:,:,-1,ind])
    densities[ind] = check_density(histories[:,:,-1,ind])

np.save("simulations/rd_10k_res1_matrices.npy", histories)
np.save("simulations/rd_10k_res1_scores.npy", scores)
np.save("simulations/rd_10k_res1_densities.npy", densities)