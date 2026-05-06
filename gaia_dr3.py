#!/usr/bin/env python3
"""
Hunt & Reffert 2023 open cluster analyzer - counts stars with 6D phase space.
Filters for clusters with >100 members with radial velocities.
"""

import subprocess
import sys
import os
from collections import defaultdict


BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/673/A114"
MIN_MEMBERS = 100


def download_file(filename):
    """Download file using curl with progress bar."""
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")
    
    result = subprocess.run(
        ["curl", "-#", "-o", filename, url],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error downloading {filename}: {result.stderr}")
        sys.exit(1)
    
    print(f"✓ Downloaded {filename}")


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


def main():
    members_file = "members.dat"
    clusters_file = "clusters.dat"
    
    # Download files if needed
    for filename in [members_file, clusters_file]:
        if not os.path.exists(filename):
            download_file(filename)
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
        if count > MIN_MEMBERS
    ]
    
    clusters.sort(key=lambda x: x['n_rv'], reverse=True)
    
    # Display
    print(f"\n{'='*80}")
    print(f"Clusters with >{MIN_MEMBERS} members with full 6D phase space")
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


if __name__ == "__main__":
    main()