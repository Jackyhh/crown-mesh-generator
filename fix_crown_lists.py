#!/usr/bin/env python3
"""
Fix crown training lists by removing non-existent entries
"""

import os

def load_available_crowns():
    """Load list of available crown directories"""
    crown_dir = "data/crown_psr/00000001"
    available = []
    for item in os.listdir(crown_dir):
        if item.startswith("crown") and os.path.isdir(os.path.join(crown_dir, item)):
            available.append(item)
    return set(available)

def fix_list_file(list_path, available_crowns):
    """Fix a list file by removing non-existent entries"""
    print(f"Fixing {list_path}...")
    
    # Read current list
    with open(list_path, 'r') as f:
        entries = [line.strip() for line in f if line.strip()]
    
    print(f"Original entries: {len(entries)}")
    
    # Filter to only existing entries
    valid_entries = [entry for entry in entries if entry in available_crowns]
    invalid_entries = [entry for entry in entries if entry not in available_crowns]
    
    print(f"Valid entries: {len(valid_entries)}")
    print(f"Invalid entries: {len(invalid_entries)}")
    if invalid_entries:
        print(f"Removing: {invalid_entries}")
    
    # Write back valid entries
    with open(list_path, 'w') as f:
        for entry in valid_entries:
            f.write(f"{entry}\n")
    
    return len(valid_entries), len(invalid_entries)

def main():
    # Get available crown directories
    available_crowns = load_available_crowns()
    print(f"Found {len(available_crowns)} available crown directories")
    
    # Fix all list files
    list_dir = "data/crown_psr/00000001"
    
    total_valid = 0
    total_invalid = 0
    
    for list_file in ["train.lst", "val.lst", "test.lst"]:
        list_path = os.path.join(list_dir, list_file)
        if os.path.exists(list_path):
            valid, invalid = fix_list_file(list_path, available_crowns)
            total_valid += valid
            total_invalid += invalid
        else:
            print(f"Warning: {list_path} not found")
    
    print(f"\nSummary:")
    print(f"Total valid entries: {total_valid}")
    print(f"Total invalid entries removed: {total_invalid}")

if __name__ == "__main__":
    main()