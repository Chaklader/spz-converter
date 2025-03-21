#!/usr/bin/env python3
"""
Calculate the centroid of a 2D polygon defined by x,z coordinates in a JSON file.
"""

import json
import numpy as np
from shapely.geometry import Polygon
import sys
import os

def calculate_centroid(json_file):
    """
    Calculate the centroid of a polygon defined by points in a JSON file.
    
    Args:
        json_file (str): Path to JSON file containing array of {x, z} coordinates
        
    Returns:
        tuple: (x, z) coordinates of the centroid
    """
    try:
        # Load the JSON data
        with open(json_file, 'r') as f:
            points_data = json.load(f)
        
        if not points_data:
            print("Error: JSON file contains no data")
            return None
        
        # Convert to numpy array of (x, z) coordinates
        points = np.array([(float(point["x"]), float(point["z"])) for point in points_data])
        
        # Method 1: Simple arithmetic mean (geometric centroid of points)
        simple_centroid = points.mean(axis=0)
        
        # Method 2: Using Shapely to calculate the true polygon centroid
        # This is more accurate for non-uniform or complex shapes
        try:
            polygon = Polygon(points)
            if polygon.is_valid:
                true_centroid = (polygon.centroid.x, polygon.centroid.y)
            else:
                print("Warning: Invalid polygon, falling back to simple centroid")
                true_centroid = simple_centroid
        except Exception as e:
            print(f"Warning: Couldn't calculate polygon centroid: {e}")
            print("Falling back to simple centroid")
            true_centroid = simple_centroid
        
        return true_centroid
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Get all JSON files in the current directory
        json_files = [f for f in os.listdir('.') if f.endswith('.json')]
        
        if not json_files:
            print("Error: No JSON files found in the current directory")
            return
        
        if len(json_files) == 1:
            json_file = json_files[0]
        else:
            print("Multiple JSON files found. Please specify which one to use:")
            for i, f in enumerate(json_files):
                print(f"{i+1}. {f}")
            
            try:
                selection = int(input("Enter the number of the file to use: "))
                json_file = json_files[selection-1]
            except (ValueError, IndexError):
                print("Invalid selection")
                return
    
    centroid = calculate_centroid(json_file)
    
    if centroid:
        print(f"Centroid of the area in {json_file}:")
        print(f"x: {centroid[0]:.4f}, z: {centroid[1]:.4f}")

if __name__ == "__main__":
    main()
