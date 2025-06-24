import os
import time
import igl

import numpy as np
from natsort import natsorted

from utils.misc import mkdir_and_rename
from utils.sm_utils import shape_loader, get_feature_opts, get_features, get_edge_cost, knn_search
from utils.shape_util import compute_geodesic_distmat, write_off, read_shape, decimate_with_mesh_lab
from utils.cycle_util import get_cycles_around_patches, optimize_patchmatch
from utils.patch_util import (fix_bridges, filter_matching, extract_patches_from_cycles, post_process_matching,
                              extract_and_save_submeshes, optimize_surfmatch)
from utils.vis_util import plot_matching, plot_dense_matching


def run_patchmatch(opt, max_depth=2):

    # load shapes
    shape_opts = {"num_faces": opt['num_faces']}
    VX, FX, vx, fx, vx2VX, VY, FY, vy, fy, vy2VY = shape_loader(opt['shape_x_file'], opt['shape_y_file'], shape_opts)

    # get distance matrix
    dist_y = compute_geodesic_distmat(vy, fy, small=opt['num_faces'] <= 1000)

    # get cycles around the patches on mesh Y
    print("Extracting poatches on mesh Y ...")
    ey, fy_assignment = get_cycles_around_patches(vy, fy, opt['k'], dist_y, desired_range=opt['desired_range'], verbose=opt['verbose'])

    # get edges of mesh x
    ex = igl.edges(fx)
    ex = np.row_stack((ex, ex[:, [1, 0]]))


    print("Extracting deep features ...")
    feature_opts = get_feature_opts(opt['dataset_name'])
    feat_x, feat_y = get_features(VX, FX, VY, FY, feature_opts, return_numpy=False)

    print("Building cost matrix ...")
    start_time_cost_matrix = time.time()
    edge_costs = get_edge_cost(feat_x, feat_y, vx2VX, vy2VY)
    end_time_cost_matrix = time.time()
    print(f"Building cost matrix took: {end_time_cost_matrix - start_time_cost_matrix}s")
    print()

    print("Running Patch-Match ...")
    matching, runtime = optimize_patchmatch(vx, vy, ex, ey, edge_costs, max_depth=max_depth)


    print("Saving intermediate results ...")
    np.save(f"{opt['output_folder']}/matching.npy", matching)
    np.save(f"{opt['output_folder']}/fy_original_assignment.npy", fy_assignment)
    np.save(f"{opt['output_folder']}/ey.npy", ey)
    np.save(f"{opt['output_folder']}/vx2VX.npy", vx2VX)
    np.save(f"{opt['output_folder']}/vy2VY.npy", vy2VY)
    np.save(f"{opt['output_folder']}/edge_costs.npy", edge_costs)
    write_off(f"{opt['output_folder']}/mesh_x.off", vx, fx)
    write_off(f"{opt['output_folder']}/mesh_y.off", vy, fy)
    print("Finished saving results.")


    print("Extracting cycles on target mesh ...")
    all_cycles_x, all_cycles_y = filter_matching(matching, ey)
    fx_assignment, fy_assignment, all_cycles_x, all_cycles_y = extract_patches_from_cycles(vx, fx, all_cycles_x,
                                                                                           vy, fy, all_cycles_y,
                                                                                           fy_assignment, matching)
    fx_assignment, all_cycles_x = post_process_matching(vx, fx, fx_assignment, all_cycles_x, all_cycles_y)
    extract_and_save_submeshes(vx, fx, fx_assignment, all_cycles_x,
                               vy, fy, fy_assignment, all_cycles_y,
                               output_folder=opt['output_folder'])

    print("Plotting the matched cycles ...")
    plot_matching(vx, fx, vy, fy, fy_assignment, fx_assignment, filename=f"{opt['output_folder']}/patch_matching.html")
    print("Done!")
    return runtime


def run_surfmatch(folder):
    exp_name = os.path.basename(folder)
    all_folders = natsorted([os.path.join(folder, f) for f in os.listdir(folder) if
                             os.path.isdir(os.path.join(folder, f)) and f.isdigit()])
    global_runtime = 0

    print(f"##################################### Pair: {exp_name} ##################################### \n")
    print(f"Number of patches: {len(all_folders)} \n")

    # load shape and pre-calculated costs
    edge_costs = np.load(f"{folder}/edge_costs.npy")
    VX, _ = read_shape(f"{folder}/mesh_x.off")
    VY, _ = read_shape(f"{folder}/mesh_y.off")

    # run patchmatch on each patch
    for i, subfolder in enumerate(all_folders):
        print(f"##################################### Pair: {exp_name} - Running patch {i}/{len(all_folders)} #####################################")

        vx, fx = read_shape(f"{subfolder}/mesh_x.off")
        vy, fy = read_shape(f"{subfolder}/mesh_y.off")
        boundary_x = np.load(f"{subfolder}/bound_x.npy")
        boundary_y = np.load(f"{subfolder}/bound_y.npy")

        vx2VX = knn_search(vx, VX)
        vy2VY = knn_search(vy, VY)

        patch_edge_costs = edge_costs[vx2VX, :][:, vy2VY]

        opts = {}
        opts["edge_cost"] = patch_edge_costs

        fx_has_to_many_faces = fx.shape[0] - fy.shape[0] > 0.05 * fy.shape[0]
        if fx_has_to_many_faces:

            vxx, fxx = decimate_with_mesh_lab(vx, fx, fy.shape[0])
            boundary_xx, dist = knn_search(vx[boundary_x[:, 0]], vxx, return_distance=True)
            assert (np.linalg.norm(dist) <= 1e-5)
            boundary_xx = np.column_stack((boundary_xx[:, None], boundary_xx[:, None]))
            boundary_xx[:-1, 1] = boundary_xx[1:, 1]
            boundary_xx[-1, 1] = boundary_xx[0, 0]
        else:
            boundary_xx = boundary_x
            vxx, fxx = vx, fx

        # fix bridges
        vxx_fixed, fxx_fixed = fix_bridges(vxx.copy(), fxx.copy(), boundary_xx)
        vy_fixed, fy_fixed = fix_bridges(vy.copy(), fy.copy(), boundary_y)
        idx_fixed = knn_search(vxx_fixed, vx)
        idy_fixed = knn_search(vy_fixed, vy)
        opts["edge_cost"] = opts["edge_cost"][idx_fixed][:, idy_fixed]

        point_map, fx_sol, fy_sol, runtime = optimize_surfmatch(vxx_fixed, fxx_fixed, None, vy_fixed, fy_fixed,
                                                                      None, boundary_xx,
                                                                      boundary_y, opts)

        # map from fixed to original mesh
        point_map[:, 0], point_map[:, 1] = idy_fixed[point_map[:, 0]], idx_fixed[point_map[:, 1]]
        fx_sol, fy_sol = idx_fixed[fx_sol], idy_fixed[fy_sol]

        global_runtime += runtime

        print("Saving patch results ...")
        # result on patch level
        np.save(f"{subfolder}/local_point_map.npy", point_map)
        np.save(f"{subfolder}/local_fx_sol.npy", fx_sol)
        np.save(f"{subfolder}/local_fy_sol.npy", fy_sol)

        # result on original mesh level
        fx_sol_global = vx2VX[fx_sol]
        fy_sol_global = vy2VY[fy_sol]
        point_map_global = np.column_stack((vy2VY[point_map[:, 0]], vx2VX[point_map[:, 1]]))

        np.save(f"{subfolder}/global_point_map.npy", point_map_global)
        np.save(f"{subfolder}/global_fx_sol.npy", fx_sol_global)
        np.save(f"{subfolder}/global_fy_sol.npy", fy_sol_global)

        print(
            f"########################################################################################################### \n")

    return global_runtime

def get_final_matching(folder):
    all_folders = natsorted([os.path.join(folder, f) for f in os.listdir(folder) if
                             os.path.isdir(os.path.join(folder, f)) and f.isdigit()])

    p2p = []
    FX_sol = []
    FY_sol = []

    VX, FX = read_shape(f"{folder}/mesh_x.off")
    VY, FY = read_shape(f"{folder}/mesh_y.off")

    for subfolder in all_folders:
        patch_p2p = np.load(f"{subfolder}/global_point_map.npy")
        patch_fx_sol = np.load(f"{subfolder}/global_fx_sol.npy")
        patch_fy_sol = np.load(f"{subfolder}/global_fy_sol.npy")

        p2p.append(patch_p2p)
        FX_sol.append(patch_fx_sol)
        FY_sol.append(patch_fy_sol)

    p2p = np.vstack(p2p)
    FX_sol = np.vstack(FX_sol)
    FY_sol = np.vstack(FY_sol)

    return VX, FX, VY, FY, p2p, FX_sol, FY_sol


def run_example():
    # Runtimes shown in this example are run on: AMD Ryzen 9 5950X 16-Core Processor with 64GB of RAM

    # load shapes
    opt = {}
    opt['k'] = 50 # Desired number of patches
    opt['desired_range'] = [10, 100] # Desired range of the number of triangles per patch
    opt['num_faces'] = 1000
    opt['dataset_name'] = "faust"
    opt['verbose'] = True
    opt['shape_x_file'] = "datasets/FAUST_r/off/tr_reg_000.off"
    opt['shape_y_file'] = "datasets/FAUST_r/off/tr_reg_001.off"
    opt['output_folder'] = f"output/example"
    mkdir_and_rename(opt['output_folder'])

    # run PatchMatch
    print("Running Patchmatch ...")
    runtime_patchmatch = run_patchmatch(opt, max_depth=2)
    print(f"Patch match took: {runtime_patchmatch}s \n") # Takes around 125s

    # run SurfMatch
    runtime_srufmatch = run_surfmatch(opt['output_folder'])
    print(f"Patch match took: {runtime_srufmatch}s \n") # Takes around 25s

    # get all matchings for vis
    print("Plotting Surfmatch results ...")
    VX, FX, VY, FY, p2p, _, _ = get_final_matching(opt['output_folder'])

    # visualize final dense results
    plot_dense_matching(VX, FX, VY, FY, p2p, dataset="faust", filename=f"{opt['output_folder']}/surf_matching.html")
    print(f"Open {opt['output_folder']}/surf_matching.html to see results")
    print("Done!")


if __name__ == "__main__":
    run_example()




