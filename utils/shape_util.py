import trimesh
import pymeshlab

import numpy as np
import open3d as o3d
import networkx as nx
from sklearn import neighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def read_shape(file, as_cloud=False):
    """
    Read mesh from file.

    Args:
        file (str): file name
        as_cloud (bool, optional): read shape as point cloud. Default False
    Returns:
        verts (np.ndarray): vertices [V, 3]
        faces (np.ndarray): faces [F, 3] or None
    """
    if as_cloud:
        verts = np.asarray(o3d.io.read_point_cloud(file).points)
        faces = None
    else:
        mesh = o3d.io.read_triangle_mesh(file)
        verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)

    return verts, faces

def compute_geodesic_distmat(verts, faces, small=False):
    """
    Compute geodesic distance matrix using Dijkstra algorithm

    Args:
        verts (np.ndarray): array of vertices coordinates [n, 3]
        faces (np.ndarray): array of triangular faces [m, 3]
        small (bool, optional): compute geodesic distance matrix for small meshes (<1000)

    Returns:
        geo_dist: geodesic distance matrix [n, n]
    """
    if small:
        loop_range = np.arange(10, 100, 10)
    else:
        loop_range = np.arange(500, 2000, 500)

    for NN in loop_range:
        # get adjacency matrix
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        vertex_adjacency = mesh.vertex_adjacency_graph
        assert nx.is_connected(vertex_adjacency), 'Graph not connected'
        vertex_adjacency_matrix = nx.adjacency_matrix(vertex_adjacency, range(verts.shape[0]))
        # get adjacency distance matrix
        graph_x_csr = neighbors.kneighbors_graph(verts, n_neighbors=NN, mode='distance', include_self=False)
        distance_adj = csr_matrix((verts.shape[0], verts.shape[0])).tolil()
        distance_adj[vertex_adjacency_matrix != 0] = graph_x_csr[vertex_adjacency_matrix != 0]
        # compute geodesic matrix
        geodesic_x = shortest_path(distance_adj, directed=False)
        if np.any(np.isinf(geodesic_x)):
            print('Inf number in geodesic distance. Increasing NN.')
        else:
            break
    if np.any(np.isinf(geodesic_x)):
        print('Inf number in geodesic distance. Increasing NN failed.')
    return geodesic_x

def write_off(file, verts, faces):
    with open(file, 'w') as f:
        f.write("OFF\n")
        f.write(f"{verts.shape[0]} {faces.shape[0]} {0}\n")
        for x in verts:
            f.write(f"{' '.join(map(str, x))}\n")
        for x in faces:
            f.write(f"{len(x)} {' '.join(map(str, x))}\n")

def decimate_with_mesh_lab(vx, fx, num_faces):
    """
        Decimate with meshlab while not changing the boundary

    Args:
        vx (np.ndarray):            array of vertices coordinates [nx, 3]
        fx (np.ndarray):            array of triangular faces [mx, 3]
        num_faces (int):            desired number of faces
    Returns:
        vxx (np.ndarray):           array of decimated vertices coordinates [nxx, 3]
        fxx (np.ndarray):           array of decimated triangular faces [mxx, 3]
    """
    m = pymeshlab.Mesh(vertex_matrix=vx, face_matrix=fx)
    ms = pymeshlab.MeshSet(verbose=False)
    ms.add_mesh(m)
    ms.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=num_faces,
        preservenormal=True,
        preservetopology=True,
        boundaryweight=1000000000.0,
        preserveboundary=True,
        optimalplacement=False,
    )
    vxx, fxx = ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()
    if vx.flags.c_contiguous: # switch to back to column major since meshlab returns in row major
        vxx, fxx = np.ascontiguousarray(vxx), np.ascontiguousarray(fxx)
    return vxx.astype(vx.dtype), fxx.astype(fx.dtype)