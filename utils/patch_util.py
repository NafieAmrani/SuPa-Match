import os
import igl
import pickle
import time
import trimesh
import numpy as np
import networkx as nx
import matplotlib
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
from scipy.linalg import orthogonal_procrustes

from utils.cycle_util import remap_values, helper_get_cycles, separate_cycles, find_inside_triangles
from utils.shape_util import write_off
from utils.sm_utils import knn_search

from surface_cycles import product_graph_generator, get_surface_cycles

def subdivide(vertices,
              faces,
              bridge_tuple):
    """
    Subdivide a mesh into smaller triangles.

    Note that if `face_index` is passed, only those
    faces will be subdivided and their neighbors won't
    be modified making the mesh no longer "watertight."

    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 3) int
      Indexes of vertices which make up triangular faces
    bridge_tuple : faces to subdivide.
      if None: all faces of mesh will be subdivided
      if (n,) int array of indices: only specified faces

    Returns
    ----------
    new_vertices : (q, 3) float
      Vertices in space
    new_faces : (p, 3) int
      Remeshed faces
    """

    face_index = np.asanyarray([bridge_tuple[0][0], bridge_tuple[1][0]])

    # the (c, 3) int array of vertex indices
    faces_subset = faces[face_index]  # (F,3)

    # find max edge of each face
    face_edges_argmax = np.asanyarray([bridge_tuple[0][1], bridge_tuple[1][1]])
    face_max_edge = np.asanyarray([list(bridge_tuple[0][2]), list(bridge_tuple[1][2])])
    assert(face_max_edge[0, 0] == face_max_edge[1, 1] and face_max_edge[0, 1] == face_max_edge[1, 0])


    # subdivide max_edge
    mid = vertices[face_max_edge[0]].mean(axis=0)[None]
    mid_idx = np.asanyarray([len(vertices), len(vertices)])

    # find another vertex of triangle out of max edge
    vertex_in_edge = np.full_like(faces_subset, fill_value=False)
    for i in range(faces_subset.shape[1]):
        for j in range(face_max_edge.shape[1]):
            vertex_in_edge[:, i] = np.logical_or(vertex_in_edge[:, i], faces_subset[:, i] == face_max_edge[:, j])
    another_vertices = faces_subset[np.logical_not(vertex_in_edge)]

    # the new faces_subset with correct winding
    f = np.column_stack([another_vertices,
                         face_max_edge[:, 0],
                         mid_idx,

                         mid_idx,
                         face_max_edge[:, 1],
                         another_vertices,
                         ]).reshape((-1, 3))
    # add new faces_subset per old face
    new_faces = np.vstack((faces, f[len(face_index):]))
    # replace the old face with a smaller face
    new_faces[face_index] = f[:len(face_index)]

    new_vertices = np.vstack((vertices, mid))

    return new_vertices, new_faces

def find_bridges_list(fx, boundary_edges_x, is_boundary_vertex_x, tri_tri_adjacency):
    bridges = []
    for f, tri in enumerate(fx):
        edge0_0 = tuple(tri[[0, 1]].tolist())
        edge0_1 = tuple(tri[[1, 2]].tolist())
        edge0_2 = tuple(tri[[2, 0]].tolist())
        neigh_tris = tri_tri_adjacency[f]
        neigh_tris = neigh_tris[neigh_tris != -1]

        for j, edg in enumerate([edge0_0, edge0_1, edge0_2]):
            if not edg in boundary_edges_x and is_boundary_vertex_x[edg[0]] and is_boundary_vertex_x[edg[1]]:
                for neigh_f in neigh_tris:
                    neigh_tri = fx[neigh_f]
                    edge_n_0 = tuple(neigh_tri[[1, 0]].tolist())
                    edge_n_1 = tuple(neigh_tri[[2, 1]].tolist())
                    edge_n_2 = tuple(neigh_tri[[0, 2]].tolist())
                    for k, edg_n in enumerate([edge_n_0, edge_n_1, edge_n_2]):
                        if edg == edg_n:
                            edg_n = (edg_n[1], edg_n[0])
                            assert(edg_n[0] == edg[1] and edg_n[1] == edg[0])
                            bridges.append(((f, j, edg), ( neigh_f, k, edg_n)))  # index in fx, edge index of f, index in adjf, edge index of adjf
                            break
    return bridges


def fix_bridges(v, f, b_match):
    is_boundary_vertex = np.zeros(v.shape[0], dtype=bool)
    is_boundary_vertex[b_match[:, 0]] = True

    boundary_edges = {}
    for e in range(len(b_match)):
        x = tuple(b_match[e, [0, 1]].tolist())
        boundary_edges[x] = e
        boundary_edges[(x[1], x[0])] = e


    i = 0
    tri_tri_adjacency, _ = igl.triangle_triangle_adjacency(f)
    bridges = find_bridges_list(f, boundary_edges, is_boundary_vertex, tri_tri_adjacency)
    max_iter = len(bridges)
    while len(bridges) > 0:
        v, f = subdivide(v, f, bridges[0])
        #ps.register_surface_mesh("xx" ,vx,fx)
        #ps.show()
        if i > max_iter:
            print("took too many iters to fix bridges")
            break
        i = i + 1
        is_boundary_vertex = np.hstack((is_boundary_vertex, np.zeros((1,), dtype=bool)))
        tri_tri_adjacency, _ = igl.triangle_triangle_adjacency(f)
        bridges = find_bridges_list(f, boundary_edges, is_boundary_vertex, tri_tri_adjacency)
    return v, f

def get_per_cycle_matches(ey, matching):
    per_cycle_matches = []
    last_index_in_matching = 0
    ey_index_in_matching = 0
    for edge_y in ey:
        if edge_y[0] == -1:
            per_cycle_matches.append(matching[last_index_in_matching:ey_index_in_matching+1, :])
            last_index_in_matching = ey_index_in_matching+1
            continue
        ey_source = edge_y[0]
        ey_target = edge_y[1]
        ey_index_in_matching = np.argwhere(
            np.logical_and(matching[:, 0] == ey_source,
                           matching[:, 1] == ey_target)
        ).flatten()[0]

    # dont forget the last cycle :)
    per_cycle_matches.append(matching[last_index_in_matching:ey_index_in_matching+1, :])
    return per_cycle_matches

def extract_cycle_nx_to_the_rescue(per_cycle_matching):
    G = nx.DiGraph()
    for match in per_cycle_matching:
        idx_x_0, idx_x_1, idx_y_0, idx_y_1 = match[2], match[3], match[0], match[1]
        G.add_edge((idx_y_0, idx_x_0), (idx_y_1, idx_x_1))
    num_cycles = 0
    for cycle in nx.simple_cycles(G):
        num_cycles += 1
    assert num_cycles == 1, f"{num_cycles} cycles found"
    cyc_x, cyc_y = [], []
    cycle.append(cycle[0])
    for i in range(1, len(cycle)):
        flip = False
        edge_x = [cycle[i-1][1], cycle[i][1]]
        is_edge_x_in_matching = np.any(((per_cycle_matching[:, [2, 3]] - np.array(edge_x)) == 0).sum(axis=1) == 2)
        if not is_edge_x_in_matching:
            edge_x = [cycle[i][1], cycle[i-1][1]]
            is_edge_x_in_matching = np.any(((per_cycle_matching[:, [2, 3]] - np.array(edge_x)) == 0).sum(axis=1) == 2)
            assert is_edge_x_in_matching
            flip = True

        cyc_x.append(edge_x)

        if flip:
            edge_y = [cycle[i][0], cycle[i - 1][0]]
        else:
            edge_y = [cycle[i - 1][0], cycle[i][0]]
        cyc_y.append(edge_y)
    return cyc_x, cyc_y

def filter_matching(matching, ey):
    per_cycle_matches = get_per_cycle_matches(ey, matching)
    all_cycles_x, all_cycles_y = [], []
    for per_cycle_match in per_cycle_matches:
        cyc_x, cyc_y = extract_cycle_nx_to_the_rescue(per_cycle_match)
        all_cycles_x.append(cyc_x)
        all_cycles_y.append(cyc_y)

    return all_cycles_x, all_cycles_y


def enclosed_patches(fx, fx_assignment, fy_assignment, sets_of_patches_requiring_merge):
    tri_tri_adjacency = igl.triangle_triangle_adjacency(fx)[0]

    if len(sets_of_patches_requiring_merge) == 0:
        return fx_assignment, fy_assignment, sets_of_patches_requiring_merge

    patches_merged = {}

    patch_ids = np.unique(fx_assignment)
    for patch_id in patch_ids:
        patch_triangles_ids = np.where(fx_assignment == patch_id)[0]
        neighboring_assignment = fx_assignment[tri_tri_adjacency[patch_triangles_ids].flatten()]
        unique, counts = np.unique(neighboring_assignment[neighboring_assignment != patch_id], return_counts=True)
        if unique.shape[0] == 1:
            new_assignment = unique[0]
            fx_assignment[fx_assignment == patch_id] = new_assignment
            fy_assignment[fy_assignment == patch_id] = new_assignment
            if patch_id in patches_merged:
                patches_merged[new_assignment] += [patch_id]
            else:
                patches_merged[new_assignment] = [patch_id]

    for k, v in patches_merged.items():
        added = False
        for id, set_to_merge in enumerate(sets_of_patches_requiring_merge):
            if k == set_to_merge[0]:
                sets_of_patches_requiring_merge[id] += v
                added = True
                break
        if not added:
            sets_of_patches_requiring_merge += [[k] + v]

    return fx_assignment, fy_assignment, sets_of_patches_requiring_merge

def check_assignment_on_source(fy, fy_assignment, fx_assignment):
    enclosed_patches = list(set(np.unique(fy_assignment).tolist()) - set(np.unique(fx_assignment).tolist()))
    tri_tri_adjacency = igl.triangle_triangle_adjacency(fy)[0]

    for patch_id in enclosed_patches:
        patch_triangles_ids = np.where(fy_assignment == patch_id)[0]
        neighboring_assignment = fy_assignment[tri_tri_adjacency[patch_triangles_ids].flatten()]
        unique, counts = np.unique(neighboring_assignment[neighboring_assignment != patch_id], return_counts=True)
        new_assignment = unique[np.argmax(counts)]
        fy_assignment[fy_assignment == patch_id] = new_assignment
    return fy_assignment, enclosed_patches

def extract_patches_from_cycles(vx, fx, cyc_x, vy, fy, cyc_y, fy_assignment, matching, small=False):
    # build dual triangle graph (triangles are nodes and are connected by adjacent edges)
    edge_triangle_adjacency_x = {}
    edge_ids = [[0, 1], [1, 2], [2, 0], [1, 0], [2, 1], [0, 2]]
    for f in range(len(fx)):
        for i in range(6):
            edge = tuple(fx[f, edge_ids[i]].tolist())
            if edge in edge_triangle_adjacency_x:
                edge_triangle_adjacency_x[edge].append(f)
            else:
                edge_triangle_adjacency_x[edge] = [f]

    G = nx.Graph()
    for edge, adjacent_tris in edge_triangle_adjacency_x.items():
        G.add_edge(adjacent_tris[0], adjacent_tris[1])

    def flatten(nested):
        for item in nested:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item

    all_boundaries_x_vert = np.array(list(flatten(cyc_x)))

    # find assignment transfer via connected components
    fx_assignment = -np.ones((len(fx),), dtype=int)
    patches_to_be_merged = {}
    num_override = 0

    for i in range(len(cyc_x)):
        GG = G.copy()
        cyc_x_edges = cyc_x[i]
        for edge in cyc_x_edges:
            if edge[0] == edge[1]:
                continue
            adjacent_tris = edge_triangle_adjacency_x[tuple(edge)]
            try:
                GG.remove_edge(adjacent_tris[0], adjacent_tris[1])
            except nx.exception.NetworkXError as e:
                continue

        for cycle_id, cc in enumerate(nx.connected_components(GG)):
            if small: # heuristic for low res shapes
                cc_to_exclude = None
                for ccx in nx.connected_components(GG):
                    verts_cc = np.unique(fx[list(ccx)]).flatten()
                    inside_elements = np.setdiff1d(verts_cc, np.array(list(flatten(cyc_x_edges))))
                    boundary_elem_inside = np.intersect1d(all_boundaries_x_vert, inside_elements)
                    if len(boundary_elem_inside) != 0:
                        cc_to_exclude = ccx.copy()
                if cc_to_exclude is None:
                    if len(cc) > 0.5 * fx.shape[0]:
                        # skipping largest components since they very likely are not the not the patch
                        continue
                else:
                    if len(cc) == len(cc_to_exclude):
                        continue
            else:
                if len(cc) > 0.5 * fx.shape[0]:
                    # skipping largest components since they very likely are not the not the patch
                    continue
            if np.any(fx_assignment[list(cc)] != -1):
                list_patches = np.unique(fx_assignment[list(cc)])
                if i in patches_to_be_merged:
                    patches_to_be_merged[i].extend(list_patches[list_patches != -1].tolist())
                else:
                    patches_to_be_merged[i] = list_patches[list_patches != -1].tolist()
                # print(f"cycle {i}: overriding")

            fx_assignment[list(cc)] = i

    # build graph which contains patch ids as nodes which are connected to adjacent, conflicting patches
    PMG = nx.Graph()  # patch merge graph
    for patch, conflicting_patches in patches_to_be_merged.items():
        for conflicting_patch in conflicting_patches:
            PMG.add_edge(patch, conflicting_patch)
    sets_of_patches_requiring_merge = []  # list of lists, each inner lists contains patches which need to be merged
    for cc in nx.connected_components(PMG):
        sets_of_patches_requiring_merge.append(list(cc))


    # merge patches
    for set_of_patches_requiring_merge in sets_of_patches_requiring_merge:
        merge_into_patch = set_of_patches_requiring_merge[0]
        for patch in set_of_patches_requiring_merge:
            fx_assignment[fx_assignment == patch] = merge_into_patch
            fy_assignment[fy_assignment == patch] = merge_into_patch


    # Check if there are any patches on X that are enclosed by another shape
    fx_assignment, fy_assignment, sets_of_patches_requiring_merge = enclosed_patches(fx, fx_assignment, fy_assignment, sets_of_patches_requiring_merge)

    cycle_ids_to_drop = []

    # get rid of patches enclosed entirely by another patch
    if np.unique(fy_assignment).shape[0] >  np.unique(fx_assignment).shape[0]:
        fy_assignment, cycle_id_to_drop = check_assignment_on_source(fy, fy_assignment, fx_assignment)
        cycle_ids_to_drop += cycle_id_to_drop


    # update cyc_x and cyc_y
    for set_of_patches_requiring_merge in sets_of_patches_requiring_merge:
        merge_into_patch = set_of_patches_requiring_merge[0]
        # update cyc_x
        new_cycle, _, _, _ = helper_get_cycles(vx, fx, fx_assignment, id=merge_into_patch)
        new_cycle_seperated_x = separate_cycles(new_cycle[0])
        assert len(new_cycle_seperated_x) == 1, f"A patch has more than one boundary on shape X -  Patch ID: {merge_into_patch}"

        new_cycle, _, _, _ = helper_get_cycles(vy, fy, fy_assignment, id=merge_into_patch)
        new_cycle_seperated_y = separate_cycles(new_cycle[0])
        assert len(new_cycle_seperated_y) == 1, f"A patch has more than one boundary on shape Y - Patch ID: {merge_into_patch}"

        new_cycle_x_igl = igl.boundary_loop(fx[fx_assignment == merge_into_patch]).tolist()
        new_cycle_y_igl = igl.boundary_loop(fy[fy_assignment == merge_into_patch]).tolist()

        e = np.zeros((len(new_cycle_y_igl), 2), dtype="int")
        e[:, 0] = new_cycle_y_igl
        e[:-1, 1] = e[1:, 0]
        e[-1, 1] = e[0, 0]

        subset_matching = matching[np.where(np.isin(matching[:, -1], set_of_patches_requiring_merge))[0]]

        ev_map, _, ef_map = igl.edge_topology(vy, fy)
        new_subset_matching = []
        for row in subset_matching:
            if row[0] == row[1]:
                new_subset_matching += [row]
                continue
            forward_index = np.where((ev_map[:, 0] == row[0]) & (ev_map[:, 1] == row[1]))[0]
            backward_index =np.where((ev_map[:, 0] == row[1]) & (ev_map[:, 1] == row[0]))[0]

            if len(forward_index) + len(backward_index) != 1:
                print("something here is not alright")

            if len(forward_index) == 1:
                index_row = forward_index[0]
            elif  len(backward_index) == 1:
                index_row = backward_index[0]
            else:
                raise Exception(f"This should never happen! {forward_index} - {backward_index} ")

            triangle_ids_next_to_edge = ef_map[index_row]
            assignment_next_to_edge = fy_assignment[triangle_ids_next_to_edge]
            if assignment_next_to_edge[0] != merge_into_patch or assignment_next_to_edge[1] != merge_into_patch:
                new_subset_matching += [row]
            a = 0
        new_subset_matching = np.array(new_subset_matching)

        # filter out rows of matchings that idx_y1 == idx_y2 and idx_1 and idx_2 appear twice and are just flipped -> causes two cycles
        indices_to_remove = []
        # Iterate over all pairs of rows
        for i in range(len(new_subset_matching)):
            for j in range(i + 1, len(new_subset_matching)):
                # Check the conditions
                if i == 0 and j == 30:
                    a = 0
                if (new_subset_matching[i, 0] == new_subset_matching[i, 1] and
                        new_subset_matching[j, 0] == new_subset_matching[j, 1] and
                        new_subset_matching[i, 0] == new_subset_matching[j, 0] and
                        new_subset_matching[i, 2] == new_subset_matching[j, 3] and
                        new_subset_matching[j, 3] == new_subset_matching[i, 2]):
                    indices_to_remove.append(j)

        final_subset_matching = []
        for i in range(len(new_subset_matching)):
            if i in indices_to_remove:
                continue
            final_subset_matching += [new_subset_matching[i]]
        final_subset_matching = np.array(final_subset_matching)
        new_cycle_x, new_cycle_y = extract_cycle_nx_to_the_rescue(final_subset_matching)

        assert np.all(np.unique(new_cycle_x_igl) in np.unique(new_cycle_x))


        cyc_x[merge_into_patch] = new_cycle_x
        cyc_y[merge_into_patch] = new_cycle_y


    # update the final cycles
    new_cyc_x = []
    new_cyc_y = []

    cycle_ids_to_drop = cycle_ids_to_drop + [item for sublist in sets_of_patches_requiring_merge for item in sublist[1:]]

    for i in range(len(cyc_x)):
        if i in cycle_ids_to_drop:
            continue
        new_cyc_x += [cyc_x[i]]
        new_cyc_y += [cyc_y[i]]

    fx_assignment = remap_values(fx_assignment)
    fy_assignment = remap_values(fy_assignment)

    return fx_assignment, fy_assignment, new_cyc_x, new_cyc_y

def filter_double_cycles(all_cycles, f_assignment, triangles):
    n = len(all_cycles)
    tri_tri_adjacency = igl.triangle_triangle_adjacency(triangles)[0]

    checked_cycles = set()

    updated_parts = []

    for i in range(n):
        cycles_patch = all_cycles[i]
        if len(cycles_patch) == 1:
            continue

        for j in range(len(cycles_patch)):
            cycle = cycles_patch[j]
            if len(cycle) <= 4 and tuple(cycle) not in checked_cycles:
                updating_info = dict()
                ids_inside_triangle = find_inside_triangles(triangles, cycle)
                if not np.isin(np.array(cycle), triangles[ids_inside_triangle]).all():
                    continue

                f_assignment_inside = f_assignment[ids_inside_triangle]
                neighbors_assignment = f_assignment[tri_tri_adjacency[ids_inside_triangle]].flatten()
                if np.all(neighbors_assignment == neighbors_assignment[0]):
                    continue

                try:
                    unique, counts = np.unique(neighbors_assignment[neighbors_assignment != f_assignment_inside[0]], return_counts=True)
                    sorted_indices = np.argsort(counts)[::-1]
                    new_assignment = unique[sorted_indices[0]]
                    if len(unique) <= 1:
                        other_possible_assignment = None
                        raise Exception("Handling this case is not implemented, not sure if it can happen in the first place.")
                    else:
                        other_possible_assignment =  unique[sorted_indices[1]]
                except:
                    raise Exception(f"No new assignment: {neighbors_assignment[neighbors_assignment != f_assignment_inside[0]]}")

                f_assignment[ids_inside_triangle] = np.full(len(ids_inside_triangle), new_assignment)
                checked_cycles.add(tuple(cycle))
                updating_info['cycle_id'] = i
                updating_info['new_assignment'] = new_assignment
                updating_info['other_possible_assignment'] = other_possible_assignment
                updating_info['cycle'] = cycle
                updated_parts += [updating_info]

    return f_assignment, updated_parts

def find_indices_of_values(array, values):
    mask = np.isin(array, values)
    indices = np.where(mask)[0]
    return indices

def update_cycles_after_post_process(updated_parts, all_cycles_x):
    for update in updated_parts:
        problematic_vertices = update['cycle']

        # update the cycle with issue
        cycle_to_update_id = update['cycle_id']
        cycle_to_update = np.array(all_cycles_x[cycle_to_update_id])[:, 0]

        # Might be an issue if value wrap around the array
        indices = find_indices_of_values(cycle_to_update, problematic_vertices)
        assert cycle_to_update[indices[0]] == cycle_to_update[indices[-1]]

        first_value = all_cycles_x[cycle_to_update_id][indices[0]][0]
        last_value = all_cycles_x[cycle_to_update_id][indices[-1]][1]

        for i, index in enumerate(indices):
            if i == len(indices) - 1:
                new_edge = [first_value, last_value]
            else:
                new_edge = [first_value, first_value]
            all_cycles_x[cycle_to_update_id][index] = new_edge

        # update the cycle with new assignment
        new_cycle_id = update['new_assignment']
        new_cycle = np.array(all_cycles_x[new_cycle_id])[:, 0]

        other_cycle_id = update['other_possible_assignment']
        other_cycle = np.array(all_cycles_x[other_cycle_id])[:, 0]

        new_indices = find_indices_of_values(new_cycle, problematic_vertices)
        other_indices = find_indices_of_values(other_cycle, problematic_vertices)

        new_values = new_cycle[new_indices]
        other_values = other_cycle[other_indices]

        assert len(new_values) >= len(other_values), f"{len(new_values)} < {len(other_values)}"
        assert new_values[0] == other_values[-1] and new_values[-1] == other_values[0], f"new_values: {new_values} - other_cycle: {other_values}"

        new_edges = []

        for i in range(len(other_indices) - 1):
            first_value = other_values[::-1][i]
            last_value = other_values[::-1][i + 1]
            new_edge = [first_value, last_value]
            new_edges += [new_edge]
            # all_cycles_x[cycle_to_update_id][index] = new_edge

        nr_new_self_edges = len(new_values) - len(other_values)
        if nr_new_self_edges > 0:
            for i in range(nr_new_self_edges):
                new_edges += [[new_values[-1] , new_values[-1]]]

        new_edges += [[new_values[-1], all_cycles_x[new_cycle_id][new_indices[-1]][1]]]

        for i, new_index in enumerate(new_indices):
            all_cycles_x[new_cycle_id][new_index] = new_edges[i]

    return all_cycles_x


def post_process_matching(vertices, triangles, f_assignment, all_cycles_x, all_cycles_y):
    cycles, _, _, _ = helper_get_cycles(vertices, triangles, f_assignment)

    # based on the candidate edges -> seperate the cycles using networkx
    all_separated_cycles = []
    for cycle_id, cycle in enumerate(cycles):
        all_separated_cycles += [separate_cycles(cycle)]

    new_f_assignment, updated_parts = filter_double_cycles(all_separated_cycles, f_assignment, triangles)
    all_cycles_x = update_cycles_after_post_process(updated_parts, all_cycles_x)

    # detect failure cases
    failing_cycles = []
    for i in range(len(all_cycles_x)):
        adj_vxi = np.zeros((vertices.shape[0], ), dtype=int)
        cyc_x_edges = all_cycles_x[i]
        for edge in cyc_x_edges:
            if edge[0] == edge[1]:
                continue
            adj_vxi[edge[0]] += 1
            adj_vxi[edge[1]] += 1
        if np.all(adj_vxi > 2):
            failing_cycles += [i]
    if len(failing_cycles) > 0:
        raise Exception(f"These cycles have some problems {failing_cycles}")

    # sanity check that cycles on both x and y are still of same shapes
    for cycle_id, cycle in enumerate(all_cycles_x):
        cycle_x = all_cycles_x[cycle_id]
        cycle_y = all_cycles_y[cycle_id]
        assert len(cycle_x) == len(cycle_y), f"Sanity check failed: {len(cycle_x)} != {len(cycle_y)}"

    new_f_assignment = remap_values(new_f_assignment)
    return new_f_assignment, all_cycles_x

def extract_mesh(f, v, f_assignment, all_cycles):
    k = np.unique(f_assignment).shape[0]
    all_patches_v = []
    all_patches_f = []
    all_boundaries = []
    for i in range(k):
        triangle_ids = np.where(f_assignment == i)[0]

        submesh = trimesh.Trimesh(vertices=np.copy(v), faces=np.copy(f), process=False)
        submesh.update_faces(triangle_ids)
        submesh.remove_unreferenced_vertices()

        vs = np.array(submesh.vertices)
        fs = np.array(submesh.faces)
        v_to_vs = knn_search(v, vs)
        boundary = v_to_vs[all_cycles[i]]

        all_patches_v += [vs]
        all_patches_f += [fs]
        all_boundaries += [boundary]



    return all_patches_v, all_patches_f, all_boundaries

def extract_and_save_submeshes(vx, fx, fx_assignment, all_cycles_x, vy, fy, fy_assignment, all_cycles_y, output_folder=None):
    if output_folder is not None:
        np.save(f"{output_folder}/fy_assignment.npy", fy_assignment)
        np.save(f"{output_folder}/fx_assignment.npy", fx_assignment)

        write_off(f"{output_folder}/mesh_x.off", vx, fx)
        write_off(f"{output_folder}/mesh_y.off", vy, fy)

        filename = f"{output_folder}/all_cycles_x.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(all_cycles_x, file)

        filename = f"{output_folder}/all_cycles_y.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(all_cycles_y, file)

    assert np.unique(fx_assignment).shape[0] == np.unique(fx_assignment).shape[0], f"Sanity check failed: {np.unique(fx_assignment).shape[0]} != {np.unique(fx_assignment).shape[0]}"
    assert np.unique(fx_assignment).shape[0] == len(all_cycles_x), f"Sanity check failed: {np.unique(fx_assignment).shape[0]} != {len(all_cycles_x)}"
    assert np.unique(fx_assignment).shape[0] == len(all_cycles_y), f"Sanity check failed: {np.unique(fy_assignment).shape[0]} != {len(all_cycles_y)}"

    all_patches_vx, all_patches_fx, boundaries_x = extract_mesh(fx, vx, fx_assignment, all_cycles_x)
    all_patches_vy, all_patches_fy, boundaries_y = extract_mesh(fy, vy, fy_assignment, all_cycles_y)

    if output_folder is not None:
        for i in range(len(all_patches_vx)):
            os.makedirs(f"{output_folder}/{i}", exist_ok=True)
            write_off(f"{output_folder}/{i}/mesh_x.off", all_patches_vx[i], all_patches_fx[i])
            write_off(f"{output_folder}/{i}/mesh_y.off", all_patches_vy[i], all_patches_fy[i])
            np.save(f"{output_folder}/{i}/bound_x.npy", boundaries_x[i])
            np.save(f"{output_folder}/{i}/bound_y.npy", boundaries_y[i])

    return all_patches_vx, all_patches_fx, all_patches_vy, all_patches_fy



def optimize_surfmatch(vx, fx, feat_x, vy, fy, feat_y, boundary_x, boundary_y, opts={}):
    """
    Solve patch using geco method while incorporating additional boudnary constraints
    ⚠️ NOTE: we use the boundary to condition the search space so that orientation flips should not happen
             => we do this by mapping both shapes to the unit circle and throw away those edge matchings which would
                result in orientation flips

    Args:
        vx (np.ndarray):            array of vertices coordinates [nx, 3]
        fx (np.ndarray):            array of triangular faces [mx, 3]
                                    asserting: all vertices of vx are used
        feat_x (np.ndarray):        array of triangular faces [nx, feat_dim]
        vy (np.ndarray):            array of vertices coordinates [ny, 3]
        fy (np.ndarray):            array of triangular faces [my, 3]
                                    asserting: all vertices of vy are used
        feat_y (np.ndarray):        array of triangular faces [ny, feat_dim]
        boundary_x (np.ndarray):    sorted array of boundary edges (forming a loop), indices into vx [bx, 2]
        boundary_y (np.ndarray):    sorted array of boundary edges (forming a loop), indices into vy [by, 2]
                                    asserting: bx == by and that each row of boundary_x corresponds to row in boundary_y
        opts (python dictionary):   additional options with potential keys
                                    - "prune": (bool), switch on/off orientation preserving pruning, default: True
                                    - "prune_hard": (bool), hard prune orientation flips, default: False
                                    - "prune_angle_thres": (int), angle threshold (degree) for pruning, default: 90
                                    - "time_limit": (int), time limit in seconds for gurobi, default: 60 * 60
                                    - "distortion_bound": (int), max num of edges-vertex maps, must be mod 2, default: 2
                                    - "edge_cost": (np.array()), |vx|x|vy| numpy array with per-vert cost, default: None

    Returns:
        matching (np.ndarray):      array of indices into vx and corresponding indices into vy [num_matches, 2]
                                    => each row are corresponding indices [idx_in_vx, idx_in_vy]
        fx_sol (np.ndarray):        array of indices into vx of matched (degenerate) triangles [num_tri_matches, 3]
        fy_sol (np.ndarray):        array of indices into vy of matched (degenerate) triangles [num_tri_matches, 3]
                                    => each row of fx_sol and fy_sol are corresponding (degnerate) triangles
    """
    ## ++++++++++++++++++++++++++++++++++++++++
    ## ++++++++ Copy options if exists ++++++++
    ## ++++++++++++++++++++++++++++++++++++++++
    debug_plotting = False
    time_limit = 60 * 60
    if "time_limit" in opts:
        time_limit = opts["time_limit"]
    prune = True
    if "prune" in opts:
        prune = opts["prune"]
    prune_hard = False
    if "prune_hard" in opts:
        prune_hard = opts["prune_hard"]
    prune_angle_thres = 45
    if "prune_angle_thres" in opts:
        prune_angle_thres = opts["prune_angle_thres"]
    assert prune_angle_thres > 0 and prune_angle_thres < 180, "prune_angle_thres should be between 0° and 180°"
    max_depth = 2
    if "distortion_bound" in opts:
        max_depth = opts["distortion_bound"]
    assert max_depth % 2 == 0, "distortion_bound must be an even number"
    edge_cost = None
    if "edge_cost" in opts:
        edge_cost = opts["edge_cost"]

    resolve_coupling = False # makes the opt problem smaller but doesnt change anything else

    ## ++++++++++++++++++++++++++++++++++++++++
    ## +++++++++++++++++ Setup ++++++++++++++++
    ## ++++++++++++++++++++++++++++++++++++++++
    ey = get_surface_cycles(fy)
    ex = igl.edges(fx)
    ex = np.row_stack((ex, ex[:, [1, 0]]))

    if edge_cost is None:
        edge_cost = np.zeros((len(vx), len(vy)))
        for i in range(0, len(vx)):
            diff = np.abs(feat_y - feat_x[i, :])
            edge_cost[i, :] = np.sum(diff, axis=1)

    ## ++++++++++++++++++++++++++++++++++++++++
    ## +++++++++++++ Setup GECO +++++++++++++++
    ## ++++++++++++++++++++++++++++++++++++++++

    pg = product_graph_generator(vx, ex, vy, ey, edge_cost)
    pg.set_resolve_coupling(resolve_coupling)
    pg.set_max_depth(max_depth)
    pg.generate()
    product_space = pg.get_product_space()

    E = pg.get_cost_vector()
    I, J, V = pg.get_constraint_matrix_vectors()
    RHS = pg.get_rhs()


    ## ++++++++++++++++++++++++++++++++++++++++
    ## ++++++++++++ gurobi setup  +++++++++++++
    ## ++++++++++++++++++++++++++++++++++++++++
    # gurobi setup & solve
    m = gp.Model("surface_cycles")
    x = m.addMVar(shape=E.shape[0], vtype=GRB.BINARY, name="x")


    ## ++++++++++++++++++++++++++++++++++++++++
    ## ++++++ Prune via mapping to disc +++++++
    ## ++++++++++++++++++++++++++++++++++++++++
    if prune:
        b_x = igl.boundary_loop(fx)
        b_y = igl.boundary_loop(fy)
        bnd_uv_x = igl.map_vertices_to_circle(vx, b_x)
        bnd_uv_y = igl.map_vertices_to_circle(vy, b_y)
        uv_x = igl.harmonic(vx, fx, b_x, bnd_uv_x, 1)
        uv_y = igl.harmonic(vy, fy, b_y, bnd_uv_y, 1)
        # best align boundaries (this should work if discretisation is similar, ie not to many edge-point matches)
        R, S = orthogonal_procrustes(uv_x[boundary_x[:, 0]], uv_y[boundary_y[:, 1]])
        uv_x = uv_x @ R

        if debug_plotting:
            import polyscope as ps
            ps.init()
            num_matches = boundary_x.shape[0]
            ED = np.column_stack((np.arange(num_matches), np.arange(num_matches)))
            ED[:-1, 1] = ED[1:, 1]
            ED = ED[:-1]
            yuv_of = np.array([0, 2])[None]
            cmap = matplotlib.colormaps['Spectral']
            rgb = cmap(np.linspace(0, 1, num_matches))[:, :3]

            ps.register_surface_mesh("uv_x", uv_x, fx)
            ps.register_surface_mesh("uv_y", uv_y + yuv_of, fy)
            cuvx = ps.register_curve_network("cuvx", uv_x[boundary_x[:, 0]], ED, material="flat")
            cuvy = ps.register_curve_network("cuvy", uv_y[boundary_y[:, 1]] + yuv_of, ED, material="flat")
            cuvx.add_color_quantity("matching", rgb, enabled=True)
            cuvy.add_color_quantity("matching", rgb, enabled=True)
            ps.show()

        # compute angle between uv coordinates of edges
        p_space_edge_vecs_y = uv_y[product_space[:, 0]] - uv_y[product_space[:, 1]]
        p_space_edge_vecs_x = uv_x[product_space[:, 2]] - uv_x[product_space[:, 3]]
        degenerate_edge_idx = np.logical_or(product_space[:, 0] == product_space[:, 1],
                                            product_space[:, 3] == product_space[:, 2])
        dot = np.sum(p_space_edge_vecs_x * p_space_edge_vecs_y, axis=1)
        nrm = np.linalg.norm(p_space_edge_vecs_x, axis=1) * np.linalg.norm(p_space_edge_vecs_y, axis=1)
        cos_angles = np.clip(dot / np.clip(nrm, 1e-8, np.inf), -1.0, 1.0)
        angles_radians = np.arccos(cos_angles)
        angles = np.degrees(angles_radians)
        angles[degenerate_edge_idx] = 0 # for degenerate edges we dont know ¯\_(ツ)_/¯


        if prune_hard:
            m.addConstr(x[angles >= prune_angle_thres] == 0, name="prune_flips")
        else:
            E[angles >= prune_angle_thres] *= E.max() * (1 + angles_radians[angles >= prune_angle_thres])[:, None]

    ## ++++++++++++++++++++++++++++++++++++++++
    ## +++++ Setup Boundary Constraints +++++++
    ## ++++++++++++++++++++++++++++++++++++++++
    k = int((max_depth / 2 + 1 ) ** 2)
    bnd_idx, dis = knn_search(np.column_stack((boundary_y, boundary_x)), product_space[:, [0, 1, 2, 3]], k=k, return_distance=True)
    assert np.linalg.norm(dis) < 1e-6, "boundary matches not found in product space"
    m.addConstr(x[bnd_idx].sum(axis=1) == 1, "fix boundary")

    ## ++++++++++++++++++++++++++++++++++++++++
    ## +++++++++++++ gurobi solve +++++++++++++
    ## ++++++++++++++++++++++++++++++++++++++++
    obj = E.transpose() @ x

    A = sp.csr_matrix((V.flatten(), (I.flatten(), J.flatten())), shape=(RHS.shape[0], E.shape[0]))
    m.addConstr(A @ x == RHS.flatten(), name="c")

    m.setObjective(obj, GRB.MINIMIZE)
    start_time = time.time()
    m.setParam('TimeLimit', time_limit)
    m.optimize()
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Optimisation took {runtime}s")
    result_vec = x.X

    ## ++++++++++++++++++++++++++++++++++++++++
    ## +++++++++++ write output +++++++++++++++
    ## ++++++++++++++++++++++++++++++++++++++++
    edge_matching = product_space[result_vec.astype('bool'), :-1] # [idx_y_0, idx_y_1, idx_x_0, idx_x_1]


    # extract tri-tri matching (doesnt work yet for everything, see output a la "sorting failed")
    f_matching = pg.convert_matching_to_surface_matching(result_vec.astype('bool'))
    fx_sol = f_matching[:, [3, 4, 5]]
    fy_sol = f_matching[:, [0, 1, 2]]
    point_map = edge_matching[:, [0, 2]]

    return point_map, fx_sol, fy_sol, runtime