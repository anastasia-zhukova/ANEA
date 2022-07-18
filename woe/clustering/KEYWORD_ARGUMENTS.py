
KEYWORD_ARGUMENTS = {
    'kMeans': [
        {},
        {
            'max_iter': 300
        }, {
            'max_iter': 1000
        }
    ],
    'hierarchical': [
        {},
        {
            'affinity': 'euclidean',
            'linkage': 'single'
        }, {
            'affinity': 'cosine',
            'linkage': 'complete'
        }
    ],
    'affinity': [
        {},
        {
            'damping': 0.5,
            'max_iter': 200,
            'verbose': True
        }, {
            'damping': 0.99,
            'max_iter': 100
        }
    ],
}


"""
kMeans default:
init="k-means++",
n_init=10,
max_iter=300,
tol=1e-4,
verbose=0,
random_state=None,
copy_x=True,
algorithm="auto",


hierarchical default:
affinity="euclidean",
memory=None,
connectivity=None,
compute_full_tree="auto",
linkage="ward",
distance_threshold=None,
compute_distances=False,


affinity default:
damping=0.5,
max_iter=200,
convergence_iter=15,
copy=True,
preference=None,
affinity="euclidean",
verbose=False,
random_state=None,
"""
