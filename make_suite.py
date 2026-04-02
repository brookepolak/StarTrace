from generate_ics import SubClusters
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_subclusters", type=int)
parser.add_argument("--seed", type=int)
parser.add_argument("--output", type=str)
parser.add_argument("--radius", type=float)

args = parser.parse_args()

sc = SubClusters(
    num_total=1000,
    num_subclusters=args.n_subclusters,
    radius=args.radius,
    subcluster_rho=1000,
    subcluster_virial_ratio=0.1, 
    global_virial_ratio=0.4,
    seed=args.seed
)

sc.save_ics(args.output)