#!/usr/bin/env python3
"""
Hunt & Reffert 2023 open cluster analyzer - counts stars with 6D phase space.
Filters for clusters with >100 members with radial velocities.
"""

import subprocess
import sys
import os
from collections import defaultdict


BASE_URL    = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/673/A114"
MIN_MEMBERS = 100
MAX_AGE     = 100

def download_and_extract(filename):
    """Download .gz file and extract it."""
    gz_file = f"{filename}.gz"
    url = f"{BASE_URL}/{gz_file}"
    
    print(f"Downloading {gz_file}...")
    result = subprocess.run(
        ["curl", "-#", "-o", gz_file, url],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error downloading {gz_file}: {result.stderr}")
        sys.exit(1)
    
    print(f"Extracting {filename}...")
    result = subprocess.run(
        ["gunzip", "-f", gz_file],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error extracting {gz_file}: {result.stderr}")
        sys.exit(1)
    
    print(f"✓ Ready: {filename}")


def count_rv_members(members_file):
    """Count members with RV for each cluster from members.dat."""
    rv_counts = defaultdict(int)
    total_counts = defaultdict(int)
    
    print(f"\nProcessing {members_file}...")
    
    with open(members_file, 'r') as f:
        for i, line in enumerate(f, 1):
            if i % 100000 == 0:
                print(f"  {i:,} stars processed...", end='\r')
            
            if len(line) < 936:
                continue
            
            cluster = line[0:20].strip()
            if not cluster:
                continue
            
            total_counts[cluster] += 1
            
            # Check if RV exists (bytes 915-936)
            rv = line[914:936].strip()
            if rv and rv != '?':
                try:
                    float(rv)
                    rv_counts[cluster] += 1
                except ValueError:
                    pass
    
    print(f"  {i:,} stars processed. Done!")
    return rv_counts, total_counts


def get_cluster_ages(clusters_file):
    """Extract ages from clusters.dat."""
    ages = {}
    
    with open(clusters_file, 'r') as f:
        for line in f:
            if len(line) < 809:
                continue
            
            cluster = line[0:20].strip()
            log_age = line[799:809].strip()
            
            if cluster and log_age:
                try:
                    ages[cluster] = 10 ** float(log_age) / 1e6  # Convert to Myr
                except ValueError:
                    pass
    
    return ages


def format_age(myr):
    """Format age in readable units."""
    if myr is None:
        return "Unknown"
    elif myr < 1:
        return f"{myr*1000:.0f} Kyr"
    elif myr < 1000:
        return f"{myr:.1f} Myr"
    else:
        return f"{myr/1000:.2f} Gyr"


def write_member_data(members_file, clusters_to_export, output_dir="gaia_data"):
    """Extract 6D phase space in cluster center-of-mass frame using astropy."""
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    # First pass: collect all members for each cluster to compute CoM
    cluster_members = {name: [] for name in clusters_to_export}
    
    print(f"\nReading member data for CoM calculation...")
    
    with open(members_file, 'r') as f:
        for i, line in enumerate(f, 1):
            if i % 200000 == 0:
                print(f"  {i:,} members scanned...", end='\r')
            
            if len(line) < 936:
                continue
            
            cluster = line[0:20].strip()
            if cluster not in clusters_to_export:
                continue
            
            try:
                # Extract phase space data
                ra = float(line[73:97].strip())
                dec = float(line[120:142].strip())
                plx = float(line[305:328].strip())
                pmra = float(line[214:237].strip())
                pmde = float(line[260:283].strip())
                rv_str = line[914:936].strip()
                
                if not rv_str or rv_str == '?' or plx <= 0:
                    continue
                
                rv = float(rv_str)
                dist = 1000.0 / plx  # pc
                
                cluster_members[cluster].append({
                    'ra': ra, 'dec': dec, 'dist': dist,
                    'pmra': pmra, 'pmde': pmde, 'rv': rv
                })
                
            except (ValueError, IndexError):
                continue
    
    print(f"\n  Collected members for {len(clusters_to_export)} clusters")
    
    # Second pass: compute CoM and write files
    print("\nComputing center-of-mass frames and writing files...")
    
    for cluster_name in clusters_to_export:
        members = cluster_members[cluster_name]
        if len(members) == 0:
            continue
        
        safe_name = cluster_name.replace(' ', '_').replace('/', '_')
        
        # Create SkyCoord for all members
        coords = SkyCoord(
            ra=[m['ra'] for m in members] * u.deg,
            dec=[m['dec'] for m in members] * u.deg,
            distance=[m['dist'] for m in members] * u.pc,
            pm_ra_cosdec=[m['pmra'] for m in members] * u.mas/u.yr,
            pm_dec=[m['pmde'] for m in members] * u.mas/u.yr,
            radial_velocity=[m['rv'] for m in members] * u.km/u.s,
            frame='icrs'
        )
        
        # Compute center of mass
        com_x = np.mean(coords.cartesian.x.to(u.pc).value)
        com_y = np.mean(coords.cartesian.y.to(u.pc).value)
        com_z = np.mean(coords.cartesian.z.to(u.pc).value)
        
        # Compute mean velocities
        v_x = coords.velocity.d_x.to(u.pc/u.Myr).value
        v_y = coords.velocity.d_y.to(u.pc/u.Myr).value
        v_z = coords.velocity.d_z.to(u.pc/u.Myr).value
        
        com_vx = np.mean(v_x)
        com_vy = np.mean(v_y)
        com_vz = np.mean(v_z)
        
        # Write cluster file in CoM frame
        with open(f"{output_dir}/{safe_name}.txt", 'w') as f:
            f.write("# x(pc) y(pc) z(pc) vx(pc/Myr) vy(pc/Myr) vz(pc/Myr)\n")
            f.write(f"# Center of mass frame for {cluster_name}\n")
            
            for i in range(len(coords)):
                # Position in CoM frame
                x = coords[i].cartesian.x.to(u.pc).value - com_x
                y = coords[i].cartesian.y.to(u.pc).value - com_y
                z = coords[i].cartesian.z.to(u.pc).value - com_z
                
                # Velocity in CoM frame
                vx = v_x[i] - com_vx
                vy = v_y[i] - com_vy
                vz = v_z[i] - com_vz
                
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {vx:.6f} {vy:.6f} {vz:.6f}\n")
    
    print(f"  Done! Wrote {len(clusters_to_export)} cluster files to {output_dir}/")


def write_cluster_info(clusters, output_file="cluster_info.txt", output_dir="gaia_data"):
    """Write cluster names and ages to file."""
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{output_file}", 'w') as f:
        f.write("# cluster_name age_Myr\n")
        for c in clusters:
            age_str = f"{c['age']:.6f}" if c['age'] else "NA"
            f.write(f"{c['name']} {age_str}\n")
    
    print(f"Wrote cluster info to {output_dir}/{output_file}")


def main():
    members_file = "members.dat"
    clusters_file = "clusters.dat"
    
    # Download and extract files if needed
    for filename in [members_file, clusters_file]:
        if not os.path.exists(filename):
            download_and_extract(filename)
        else:
            print(f"✓ Found {filename}")
    
    # Count RV members
    rv_counts, total_counts = count_rv_members(members_file)
    
    # Get ages
    ages = get_cluster_ages(clusters_file)
    
    # Filter and sort
    clusters = [
        {
            'name': name,
            'n_rv': count,
            'n_total': total_counts[name],
            'age': ages.get(name)
        }
        for name, count in rv_counts.items()
        if count > MIN_MEMBERS and ages.get(name) < MAX_AGE
    ]
    
    clusters.sort(key=lambda x: x['n_rv'], reverse=True)
    
    # Display
    print(f"\n{'='*80}")
    print(f"Young clusters (<{MAX_AGE} Myr) with >{MIN_MEMBERS} members with full 6D phase space")
    print(f"Hunt & Reffert 2023 (Gaia DR3) - {len(clusters)} clusters found")
    print(f"{'='*80}\n")
    
    print(f"{'Cluster':<25} {'6D Members':>12} {'Total':>10} {'Age':>15}")
    print(f"{'-'*25} {'-'*12:>12} {'-'*10:>10} {'-'*15:>15}")
    
    for c in clusters:
        print(f"{c['name']:<25} {c['n_rv']:>12,} {c['n_total']:>10,} {format_age(c['age']):>15}")
    
    # Stats
    total_6d = sum(c['n_rv'] for c in clusters)
    with_age = [c for c in clusters if c['age']]
    
    print(f"\n{'='*80}")
    print(f"Total 6D members: {total_6d:,}")
    print(f"Average per cluster: {total_6d/len(clusters):.1f}")
    
    if with_age:
        avg_age = sum(c['age'] for c in with_age) / len(with_age)
        youngest = min(with_age, key=lambda x: x['age'])
        oldest = max(with_age, key=lambda x: x['age'])
        
        print(f"\nAge range: {format_age(youngest['age'])} to {format_age(oldest['age'])}")
        print(f"Average age: {format_age(avg_age)}")
    
    print(f"{'='*80}\n")
    
    # Write output files
    cluster_names = [c['name'] for c in clusters]
    write_member_data(members_file, cluster_names)
    write_cluster_info(clusters)
    
    print("\nDone! Check gaia_data/ directory for individual cluster files.")



if __name__ == "__main__":
    main()