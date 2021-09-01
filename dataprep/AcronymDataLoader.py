import os
import numpy as np
import pickle
import sys
import json
import trimesh

from torch.utils.data import Dataset
from acronym_tools import load_mesh, load_grasps

def pc_normalize(pc):
    centroid = np.mean(pc, axis = 0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc/m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ACRONYMDataLoader(Dataset):
    def __init__(self, root, split="graps"):
        self.root = root
        self.split = split
        self.mesh_root = os.path.join(self.root, 'meshes')
        self.grasps_root = os.path.join(self.root, 'grasps')

        self.datapath = [file for file in os.listdir(self.grasps_root)]
        
    def _get_item(self, index):
        if self.split=="grasps":
            obj_mesh = load_mesh(index, mesh_root_dir=self.mesh_root)

            T, success = load_grasps(index)

            return obj_mesh, T
    
    def __getitem__(self, index):
        return self._get_item(index)

if __name__ == '__main__':
    import torch

    data = ACRONYMDataLoader('./data/acronym/grasps', split='grasps')
    print(data.__getitem__(0))
    #DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    #for point, label in DataLoader:
    #    print(point.shape)
    #    print(label.shape)