import time
import igl

import numpy as np
import gurobipy as gp
import networkx as nx
from gurobipy import GRB
import scipy.sparse as sp


from utils.fps_approx import farthest_point_sampling_distmat_boundary


from surface_cycles import product_graph_generator


def optimize_patchmatch(vx, vy, ex, ey, edge_costs, max_depth=2):
    time_limit = 60 * 60 # 1h budget
    resolve_coupling = False  # makes the opt problem smaller but doesnt change anything else
    avoid_cycle_self_intersections = True
    if avoid_cycle_self_intersections:
        print("[GECO] Avoiding self intersections of cycles")
    ## ++++++++++++++++++++++++++++++++++++++++
    ## +++++++ Solve with SurfaceCycles +++++++
    ## ++++++++++++++++++++++++++++++++++++++++
    pg = product_graph_generator(vx, ex, vy, ey, edge_costs)
    pg.set_resolve_coupling(resolve_coupling)
    pg.set_max_depth(max_depth)
    pg.generate()
    product_space = pg.get_product_space()

    E = pg.get_cost_vector()
    I, J, V = pg.get_constraint_matrix_vectors()
    RHS = pg.get_rhs()

    # gurobi setup & solve
    m = gp.Model("surface_cycles")

    # use concurrent
    print("[GECO] Using method 3")
    m.setParam("Method", 3)

    x = m.addMVar(shape=E.shape[0], vtype=GRB.BINARY, name="x")
    obj = E.transpose() @ x

    A = sp.csr_matrix((V.flatten(), (I.flatten(), J.flatten())), shape=(RHS.shape[0], E.shape[0]))
    m.addConstr(A @ x == RHS.flatten(), name="c")
    if avoid_cycle_self_intersections:
        Ileq, Jleq, Vleq = pg.get_constraint_matrix_vectors_intersections()
        Aleq = sp.csr_matrix((Vleq.flatten(), (Ileq.flatten(), Jleq.flatten())), shape=(Ileq.max()+1, E.shape[0]))
        m.addConstr(Aleq @ x <= 1, name="avoid_intersections")

    m.setObjective(obj, GRB.MINIMIZE)

    start_time = time.time()
    m.setParam('TimeLimit', time_limit)
    m.optimize()
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Optimisation took {runtime}s")
    result_vec = x.X

    result_vec = np.round(result_vec).astype("bool")
    matching = product_space[result_vec]

    return  matching, runtime

def assign_triangles_to_clusters(triangles, vertex_assignments):

    # Get the cluster assignments for each vertex in the triangles
    triangle_clusters = vertex_assignments[triangles]

    # Check if at least two vertices are in the same cluster
    unique_counts = np.array([np.bincount(t, minlength=vertex_assignments.max()+1) for t in triangle_clusters])
    majority_clusters = np.argmax(unique_counts, axis=1)
    majority_counts = np.max(unique_counts, axis=1)

    # Initialize assignments with majority clusters where applicable
    triangle_assignments = np.where(majority_counts >= 2, majority_clusters, -1)

    # Handle triangles where all vertices are in different clusters
    different_clusters = (triangle_assignments == -1)
    count = 10
    tri_tri_adjacency = igl.triangle_triangle_adjacency(triangles)[0]
    while np.any(different_clusters) and count > 0:
        unknown = np.where(different_clusters)[0]
        for u in unknown:
            for j in tri_tri_adjacency[u]:
                if not j in unknown:
                    triangle_assignments[u] = triangle_assignments[j]
                    break
        different_clusters = (triangle_assignments == -1)
        count -= 1
    if count == 0:
        raise Exception("Infinite loop in clustering triangles")

    # straighten the patches
    for triangle_id, triangle in enumerate(triangles):
        neigh_triangle_ids = tri_tri_adjacency[triangle_id]
        current_assignment = triangle_assignments[triangle_id]
        neigh_assignment = triangle_assignments[neigh_triangle_ids]
        unique, counts = np.unique(neigh_assignment, return_counts=True)
        for idx, count in enumerate(counts):
            if count == 2:
                triangle_assignments[triangle_id] = unique[idx]

    if count == 0:
        raise Exception("Detected infinite loop!")

    return triangle_assignments

def create_clusters(dist, center_ids):
    distances_to_selected = dist[:, center_ids]
    closest_indices_in_selected = np.argmin(distances_to_selected, axis=1)
    assignments = center_ids[closest_indices_in_selected]
    return assignments

def make_smaller_patches(vertices, triangles, f_assignment, dist, desired_range, verbose):
    unique_assignmnents, counts = np.unique(f_assignment, return_counts=True)
    additional_k_counter = unique_assignmnents.shape[0]

    ids_big_patches = np.where(counts >= desired_range[1])[0]
    ids_small_patches = np.where(counts < desired_range[0])[0]

    if len(ids_big_patches) +  len(ids_big_patches) == 0:
        if verbose:
            print(f"All patches have {desired_range} faces - Check again: False")
        return f_assignment, False

    # divide big patches
    for id in ids_big_patches:
        nr_faces = np.where(f_assignment == id)[0].shape[0]

        # split at least into two parts
        mini_k = max(int(nr_faces/desired_range[1]), 2)

        patch_triangles = triangles[f_assignment == id]
        patch_vertices_ids = np.unique(patch_triangles)

        index_mapping_traingles_in_patch = {old_idx: new_idx for new_idx, old_idx in enumerate(patch_vertices_ids)}
        patch_triangles = np.array([[index_mapping_traingles_in_patch[i] for i in triangle] for triangle in patch_triangles])

        # sample centroids of two new patches
        dist_in_patch = dist[np.ix_(patch_vertices_ids, patch_vertices_ids)]
        sampled_indices = farthest_point_sampling_distmat_boundary(dist_in_patch, mini_k, random_init=False)[1:]

        # cluster inside triangles with the labels of the newly sampled centroids
        v_assignment_in_patch = create_clusters(dist_in_patch, sampled_indices)
        f_assignment_in_patch = assign_triangles_to_clusters(patch_triangles, v_assignment_in_patch)
        f_assignment_in_patch = remap_values(f_assignment_in_patch)
        f_assignment[np.where(f_assignment == id)[0]] = f_assignment_in_patch + additional_k_counter
        additional_k_counter += mini_k

    # combine small patches
    tri_tri_adjacency = igl.triangle_triangle_adjacency(triangles)[0]
    for id in ids_small_patches:
        patch_triangles_ids = np.where(f_assignment == id)[0]
        neighbors_assignment = f_assignment[tri_tri_adjacency[patch_triangles_ids]].flatten()

        try:
            new_assignment = np.argmax(np.bincount(neighbors_assignment[neighbors_assignment != id]))
        except:
            raise Exception(f"No new assignment: {neighbors_assignment[neighbors_assignment != id]}")

        f_assignment[patch_triangles_ids] = np.full(len(patch_triangles_ids), new_assignment)

    # debug_plot(vertices, triangles, f_assignment)

    if verbose:
        unique_assignmnents, counts = np.unique(f_assignment, return_counts=True)
        ids_big_patches = np.where(counts >= desired_range[1])[0]
        ids_small_patches = np.where(counts >= desired_range[1])[0]
        print(f"Nr of big patches {len(ids_big_patches)} - Nr of smaller patches {len(ids_small_patches)}")
    return remap_values(f_assignment), True

def remap_values(array):
    unique_values = np.unique(array)
    unique_values = unique_values[unique_values != -1]
    value_map = {value: i for i, value in enumerate(unique_values) if value != -1}
    remapped_array = np.array([value_map.get(value, -1) for value in array.flatten()]).reshape(array.shape)
    return remapped_array

def small_patches_merging(vertices, triangles, f_assignment, min_num_faces):
    patch_ids = np.unique(f_assignment)
    tri_tri_adjacency = igl.triangle_triangle_adjacency(triangles)[0]

    G = nx.Graph()

    small_graph_patches = nx.Graph()

    special_ids = []

    for patch_id in patch_ids:
        patch_triangle_ids = np.where(f_assignment == patch_id)[0]
        if patch_triangle_ids.shape[0] < min_num_faces:
            special_ids += [patch_id]
        neighboring_assignment = f_assignment[tri_tri_adjacency[patch_triangle_ids]]
        neighboring_patches = neighboring_assignment.flatten()

        for neighboring_patch in neighboring_patches:
            if neighboring_patch != patch_id:
                G.add_edge(patch_id, neighboring_patch)
                if patch_id in special_ids and np.where(f_assignment == neighboring_patch)[0].shape[0] < min_num_faces:
                    small_graph_patches.add_edge(patch_id, neighboring_patch)

    if len(special_ids) == 0:
        return f_assignment

    nodes_to_merge = list(nx.connected_components(small_graph_patches))
    new_patches = []
    counter_ids = np.unique(f_assignment).shape[0]

    for nodes in nodes_to_merge:
        new_patch_id = counter_ids
        counter_ids += 1
        for patch_id in nodes:
            for neighbor in list(G.neighbors(patch_id)):
                if neighbor != new_patch_id:
                    G.add_edge(new_patch_id, neighbor)
                    if neighbor in nodes:
                        f_assignment[f_assignment == neighbor] = new_patch_id
        for patch_id in nodes:
            G.remove_node(patch_id)
        new_patches += [new_patch_id]

    for new_patch_id in new_patches:
        updated = False
        num_patches = np.unique(f_assignment).shape[0]
        GG = G.copy()
        GG.remove_node(new_patch_id)

        connected_componenets = list(nx.connected_components(GG))
        for cc  in connected_componenets:
            if len(cc) >= 0.5 * num_patches:
                continue
            for p in cc:
                f_assignment[f_assignment == p] = new_patch_id
                updated = True
        # case when the merged patch doesn't disconnect the graph -> merge with the smallest nearest patch
        if not updated:
            smallest_neigh_id = -1
            smallest_neigh_num_faces = np.inf
            for neighbor in list(G.neighbors(new_patch_id)):
                num_tri_neigh = np.where(f_assignment == neighbor)[0].shape[0]
                if smallest_neigh_num_faces > num_tri_neigh:
                    smallest_neigh_id = neighbor
                    smallest_neigh_num_faces = num_tri_neigh

            assert smallest_neigh_id != -1
            f_assignment[f_assignment == smallest_neigh_id] = new_patch_id

    return remap_values(f_assignment)

def process_nr_faces(vertices, triangles, f_assignment, dist, desired_range, verbose):
    # make sure all patches have a lower number of faces than desired_nr_faces
    check_patch_size = True
    counter = 0
    while check_patch_size and counter < 10:
        f_assignment, check_patch_size = make_smaller_patches(vertices, triangles, f_assignment, dist, desired_range=desired_range, verbose=verbose)
        counter += 1

    if counter == 10:
        raise Exception("Detected infinite loop in subdividing the patches")

    f_assignment = remap_values(f_assignment)
    f_assignment = small_patches_merging(vertices, triangles, f_assignment, min_num_faces=desired_range[0])
    f_assignment = remap_values(f_assignment)
    return f_assignment

def helper_get_cycles(vertices, triangles, f_assignment, id=None):
    ev_map, _, ef_map = igl.edge_topology(vertices, triangles)
    ef_assignment = f_assignment[ef_map]
    edges_between_patches_indices = np.where(ef_assignment[:, 0] != ef_assignment[:, 1])[0]

    if id is None:
        k = np.unique(f_assignment).shape[0]
        loop = range(k)
    else:
        loop = [id]

    cycles = []
    for i in loop:
        current_cycle_assginement = (ef_assignment[edges_between_patches_indices, 0] == i) | (
                    ef_assignment[edges_between_patches_indices, 1] == i)
        current_cycle_assginement_indices = edges_between_patches_indices[current_cycle_assginement]
        edges_current_cycle = ev_map[current_cycle_assginement_indices]
        cycles += [edges_current_cycle]

    return cycles, ev_map, ef_map, ef_assignment

def separate_cycles(cycle):
    G = nx.Graph()
    G.add_edges_from(cycle)
    cycles = list(nx.cycle_basis(G))
    return cycles

def find_inside_triangles(triangles, indices):
    indices_set = set(indices)
    matching_triangles = []
    for i, triangle in enumerate(triangles):
        if len(set(triangle) & indices_set) == 3:
            matching_triangles.append(i)
    return matching_triangles

def filter_short_cycles(all_cycles, f_assignment, triangles, verbose=False):
    n = len(all_cycles)
    tri_tri_adjacency = igl.triangle_triangle_adjacency(triangles)[0]

    checked_cycles = set()

    # case when a patch is all enclosed in another shape
    for i in range(n):
        sublist_A = all_cycles[i]
        for j in range(len(sublist_A)):
            cycle = sublist_A[j]
            if len(cycle) <= 4 and tuple(cycle) not in checked_cycles:
                ids_inside_triangle = find_inside_triangles(triangles, cycle)

                if not np.isin(np.array(cycle), triangles[ids_inside_triangle]).all():
                    continue
                f_assignment_inside = f_assignment[ids_inside_triangle]
                neighbors_assignment = f_assignment[tri_tri_adjacency[ids_inside_triangle]].flatten()
                if np.all(neighbors_assignment == neighbors_assignment[0]):
                    continue

                try:
                    new_assignment = np.argmax(np.bincount(neighbors_assignment[neighbors_assignment != f_assignment_inside[0]]))
                except:
                    raise Exception(f"No new assignment: {neighbors_assignment[neighbors_assignment != f_assignment_inside[0]]}")

                f_assignment[ids_inside_triangle] = np.full(len(ids_inside_triangle), new_assignment)
                checked_cycles.add(tuple(cycle))

    if verbose:
        print("Cycles with less than 4:", len(checked_cycles), "Update:", len(checked_cycles) != 0)
    return f_assignment, len(checked_cycles) != 0

def merge_patch_with_neighbor(triangles, f_assignment, cycle_id, levels=1):
    tri_tri_adjacency = igl.triangle_triangle_adjacency(triangles)[0]

    for level in range(levels):
        patch_triangle_ids = np.where(f_assignment == cycle_id)[0]
        neighbors_assignment = f_assignment[tri_tri_adjacency[patch_triangle_ids]].flatten()
        unique_neighbors = np.unique(neighbors_assignment)
        all_patch_ids_to_merge = unique_neighbors.tolist()

        for neighbor_id in unique_neighbors:
            neigh_patch_triangle_ids = np.where(f_assignment == neighbor_id)[0]
            neigh_neigh_assignment = f_assignment[tri_tri_adjacency[neigh_patch_triangle_ids]].flatten()
            neigh_neigh = np.unique(neigh_neigh_assignment)
            all_patch_ids_to_merge += neigh_neigh.tolist()

        all_patch_ids_to_merge = np.unique(np.array(all_patch_ids_to_merge))

        for patch_id in all_patch_ids_to_merge:
            f_assignment[f_assignment == patch_id] = cycle_id

    return f_assignment

def check_size_patches(f_assignment, desired_range, verbose):
    unique_assignmnents, counts = np.unique(f_assignment, return_counts=True)

    ids_big_patches = np.where(counts >= desired_range[1])[0]
    ids_small_patches = np.where(counts < desired_range[0])[0]

    if len(ids_big_patches) + len(ids_small_patches) == 0:
        if verbose:
            print(f"Final check: All patches have {desired_range} faces - Nr. small {len(ids_small_patches)} - Nr. big {len(ids_big_patches)}")
    else:
        if verbose:
            print(f"Final check: Not all patches have {desired_range} faces - Nr. small {len(ids_small_patches)} - Nr. big {len(ids_big_patches)}")

def adapt_cycle(all_cycles, triangles, f_assignment):
    final_e = None
    for i in range(len(all_cycles)):
        try:
            bloop = igl.boundary_loop(triangles[f_assignment == i])
        except:
            print(f"Problem with cycle: {i}")
        e = np.zeros((len(bloop), 2), dtype="int")
        e[:, 0] = bloop
        e[:-1, 1] = e[1:, 0]
        e[-1, 1] = e[0, 0]

        if final_e is None:
            final_e = e
        else:
            final_e = np.row_stack((final_e, np.array([-1, -1]), e))
    return final_e

def get_cycles_around_patches(vertices, triangles, k, dist, desired_range, verbose=False, resets=0):
    if resets > 5:
        raise Exception("Resets cannot be greater than 5 - Cycle generation failed!")

    original_k = k
    print(f"Start creating patches with {k} patches ...")
    start_time = time.time()

    centers_y_ids = farthest_point_sampling_distmat_boundary(dist, k, random_init=False)
    v_assignment = create_clusters(dist, centers_y_ids)
    f_assignment = assign_triangles_to_clusters(triangles, v_assignment)
    f_assignment = remap_values(f_assignment)


    # continue subdividing patches until there are no patches with more than one boundary
    lists_with_more_than_one_cycle = 0
    subdivide = True
    count = 0
    nr_update = 0
    while subdivide and count <= 50 and nr_update < 3:
        count += 1
        subdivide = False

        # make sure all patches have nr. faces in the desired range
        f_assignment = process_nr_faces(vertices, triangles, f_assignment, dist, desired_range, verbose=verbose)

        k = np.unique(f_assignment).shape[0]
        additional_k_counter = k + 1

        # get candidate edges that are in between two triangles assigned to two different clusters
        cycles, _, _, _ = helper_get_cycles(vertices, triangles, f_assignment)

        # based on the candidate edges -> seperate the cycles using networkx
        all_separated_cycles = []
        for cycle_id, cycle in enumerate(cycles):
            all_separated_cycles += [separate_cycles(cycle)]

        # filter out very short cycles (small patches enclosed in bigger ones - heuristic)
        f_assignment, updated = filter_short_cycles(all_separated_cycles, f_assignment, triangles, verbose=verbose)

        if updated:
            subdivide = True
        else:
            if verbose:
                lists_with_more_than_one_cycle = sum(1 for sublist in all_separated_cycles if len(sublist) != 1)
                print(f"Patches to subdivide: {lists_with_more_than_one_cycle}/{len(all_separated_cycles)} "
                      f"- Unique f_assignment: {np.unique(f_assignment).shape[0]}")

            for cycle_id, separated_cycles in enumerate(all_separated_cycles):
                # create 2 more patches inside this patch
                if len(separated_cycles) > 1:

                    subdivide = True
                    mini_k = 2
                    patch_triangles = triangles[f_assignment == cycle_id]
                    boundary_vertices = [elem for cycle in separated_cycles for elem in cycle]
                    patch_vertices_ids = np.unique(patch_triangles)

                    # debug_plot(vertices, triangles, f_assignment=f_assignment, patch_vertices_ids=patch_vertices_ids)
                    if len(patch_vertices_ids) < len(np.unique(boundary_vertices)):
                        triangles_with_boundary_vertices = find_inside_triangles(triangles, boundary_vertices)
                        f_assignment[triangles_with_boundary_vertices] = cycle_id
                        continue

                    # problematic patches -> merge them with biggest neighboring patch and check again
                    if count > 30:
                        count = 0
                        nr_update += 1
                        for inside_cycle_id, inside_seperated_cycles in enumerate(all_separated_cycles):
                            if len(inside_seperated_cycles) <= 1:
                                continue
                            f_assignment = merge_patch_with_neighbor(triangles, f_assignment, inside_cycle_id)
                        if verbose:
                            print(f"-------------------------------------------------------- Patch updated {nr_update} --------------------------------------------------------")
                        break

                    index_mapping_traingles_in_patch = {old_idx: new_idx for new_idx, old_idx in enumerate(patch_vertices_ids)}
                    patch_triangles = np.array([[index_mapping_traingles_in_patch[i] for i in triangle] for triangle in patch_triangles])

                    # sample centroids of two new patches
                    dist_in_patch = dist[np.ix_(patch_vertices_ids, patch_vertices_ids)]
                    mask = np.isin(patch_vertices_ids, boundary_vertices)
                    points_to_avoid = np.where(mask)[0]
                    sampled_indices = farthest_point_sampling_distmat_boundary(dist_in_patch, mini_k, points_to_avoid)

                    # cluster inside triangles with the labels of the newly sampled centroids
                    v_assignment_in_patch = create_clusters(dist_in_patch, sampled_indices)
                    f_assignment_in_patch = assign_triangles_to_clusters(patch_triangles, v_assignment_in_patch)
                    f_assignment_in_patch = remap_values(f_assignment_in_patch)
                    f_assignment[np.where(f_assignment == cycle_id)[0]] = f_assignment_in_patch + additional_k_counter
                    additional_k_counter += 2

        f_assignment = remap_values(f_assignment)



    # sanity check of the size of the patches
    if verbose:
        check_size_patches(f_assignment, desired_range, verbose=verbose)

    if nr_update == 3:
        check_even_or_odd = lambda number: -number if number % 2 == 0 else number
        new_k = original_k + check_even_or_odd(resets + 1) * 10
        if verbose:
            print(f"--------------------------------------------- Reset {resets + 1} - New_k {new_k} ---------------------------------------------")
        result = get_cycles_around_patches(vertices, triangles, new_k, dist, desired_range, verbose=verbose, resets=resets + 1)
        return result

    if count == 51:
        if verbose:
            print(" =================================================================== Stopped ======================================================================================================================================")
        raise Exception("Patch generation went into an infinite loop. Please send the vertices, triangle and nr_patches to Nafie to check.")

    assert lists_with_more_than_one_cycle == 0, f"Nr. patches with more than one cycles: {lists_with_more_than_one_cycle}"

    adapted_cycles = adapt_cycle(all_separated_cycles, triangles, f_assignment)

    end_time = time.time()
    print(f"Creating cycles took: {end_time - start_time}s")

    return adapted_cycles, f_assignment