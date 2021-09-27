from dataprep.AcronymDataLoader import ACRONYMDataLoader
import torch
import argparse
import os
from model.model import PCAutoEncoder
from model.model_fxia22 import PointNetAE
from matplotlib import pyplot as plt
import numpy as np
from dataprep.AcronymDataLoader import ACRONYMDataLoader
import trimesh
from trimesh.scene import split_scene

parser = argparse.ArgumentParser()

parser.add_argument("--input_folder", required=True, help="Single 3d model or input folder containing 3d models")
parser.add_argument("--nn_model", required=True, help="Trained Neural Network Model")
parser.add_argument("--nn_model_type", required=True, choices=['fxia', 'dhiraj'], help="Model Type")
parser.add_argument("--num_points", required=True, type=int, help="Number of points")
parser.add_argument("--out_norm_input", action="store_true", help="Output normalized version of input file")
parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')


ip_options = parser.parse_args()
input_folder = ip_options.input_folder

# Setting the Gradient Calculation Feature off
# torch.set_grad_enabled(False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

point_dim = 3
num_points = ip_options.num_points

if ip_options.nn_model_type == 'dhiraj':
    autoencoder = PCAutoEncoder(point_dim, num_points)
elif ip_options.nn_model_type == 'fxia':
    autoencoder = PointNetAE(num_points)

state_dict = torch.load(ip_options.nn_model, map_location=device)
autoencoder.load_state_dict(state_dict)

       
def infer_model_index(input_folder, ip_options, autoencoder):
    data = ACRONYMDataLoader(root=input_folder, split='meshes', num_points=ip_options.num_points)
    
    for iter in range(len(data)):
        idx_choice = np.random.choice(len(data))
        point_test, point_label = data.__getitem__(idx_choice)
        # extract only "N" number of point from the Point Cloud
        choice = np.random.choice(len(point_test), num_points, replace=True)
        point_test = point_test[choice, :]
        '''for i in range (len(point_test)):
            if point_test[i,0]<0:
                point_test[i,0] = 0
                point_test[i,1] = 0
                point_test[i,2] = 0'''
        point_test_saved = point_test
        point_label = point_label[choice, :]
        point_test = torch.from_numpy(point_test).float()
        point_test = torch.unsqueeze(point_test, 0) #done to introduce batch_size of 1 
        point_test = point_test.transpose(2, 1)
        #point_test = point_test.cuda() #uncomment this if running on GPU
        autoencoder = autoencoder.eval()
        reconstructed_points, global_feat = autoencoder(point_test)
        #reconstructed_points = autoencoder(points)

        #Reshape 
        reconstructed_points = reconstructed_points.squeeze().transpose(0,1)
        reconstructed_points = reconstructed_points.numpy()

        '''fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(1, 2, 1 , projection="3d")
        ax.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], reconstructed_points[:, 2])
        ax.set_title("Reconstructed Points") 
        ax = fig.add_subplot(1, 2, 2 , projection="3d")
        ax.scatter(point_label[:, 0], point_label[:, 1], point_label[:, 2])
        ax.set_title("True Points")
        plt.show()'''
        
        max_x = np.max(point_label[:,0], axis=0)
        print('max x: ', max_x)
        point_mesh = trimesh.PointCloud(point_test_saved, colors=[255, 0, 0, 255]) #point test is red
        reconstructed_points = reconstructed_points + np.array([2.5*max_x, 0, 0])
        reconstructed_mesh = trimesh.PointCloud(reconstructed_points, colors=[0, 255, 0, 255]) #point reconstruct is greedn

        axis = trimesh.creation.axis(origin_color= [1., 0, 0])

        scene = trimesh.Scene()
        scene.add_geometry(point_mesh)
        scene.add_geometry(reconstructed_mesh)
        scene.add_geometry(axis)
        scene.show(background=[0, 0 , 0 ,0])


with torch.no_grad():
    infer_model_index(input_folder, ip_options, autoencoder)
