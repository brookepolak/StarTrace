from generate_ics import SubClusters

N_stars = 1000
N_subcs = 10
Radius  = 12.0
rho_subc  = 1000
seed = 42

sc = SubClusters(N_stars, N_subcs, Radius, rho_subc, 
                 subcluster_virial_ratio=0.1, 
                 global_virial_ratio=0.4, seed=seed)

# # Create subclusters with internal virial ratio 0.1 and global virial ratio 0.3
# ic = SubClusters(
#     num_total=10000,
#     num_subclusters=10,
#     radius=10.0,           # pc
#     subcluster_radii=1.0,  # pc
#     subcluster_virial_ratio=0.1,  # cold subclusters
#     global_virial_ratio=0.3,       # collapsing system
#     masses=1.0,            # Msun
#     seed=42
# )
sc.check_virial()
sc.plot_radial_density_profile()
sc.plot_vel_ics()
sc.save_ics()


# My suite will be: 1000 simulations, 100 each with N=1,2,…,9,10 initial subclusters. Radius=10. seed different each time. I will be running on snellius. I want a script setup_sims.sh that sets up a bunch of directorys and puts initial conditions for each sim in a correct one. naming paradigm: SC1S10 for n subcluster and random seed. 
# then i want a submission script that simultaneously runs each simulation on one CPU. The calls for each sim will be:
# petar.init ics.txt (to make ICs for petar)
# petar -u 1 -r 0.5 -t 500 ics.txt.input