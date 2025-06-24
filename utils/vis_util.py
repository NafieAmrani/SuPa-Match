import pyvista as pv
import numpy as np
import colorcet as cc

def sample_colors(n, pad=0):
    return np.array((cc.glasbey_bw * 5))[pad:pad + n]

def create_pv_mesh(V, F):
    F = np.hstack([np.full((F.shape[0], 1), 3), F]).flatten()
    mesh = pv.PolyData(V, F)
    return mesh

def rotate_mesh(mesh, degree_x=None, degree_y=None, degree_z=None):
    if degree_x is not None:
        mesh = mesh.rotate_x(degree_x)
    if degree_y is not None:
        mesh = mesh.rotate_y(degree_y)
    if degree_z is not None:
        mesh = mesh.rotate_z(degree_z)
    return mesh

def create_pyvista_arrow(point1, point2, scale=0.01):
    direction = point2 - point1
    return pv.Arrow(point1, direction=direction,scale=scale)

def create_colormap(contour):
    minx = contour[:, 0].min()
    miny = contour[:, 1].min()
    minz = contour[:, 2].min()
    maxx = contour[:, 0].max()
    maxy = contour[:, 1].max()
    maxz = contour[:, 2].max()
    r = (contour[:, 0] - minx) / (maxx - minx)
    g = (contour[:, 1] - miny) / (maxy - miny)
    b = (contour[:, 2] - minz) / (maxz - minz)
    colors = np.stack((1-r, 1-g, b), axis=-1)

    assert colors.shape == contour.shape
    return colors


def plot_matching(VX, FX, VY, FY, fy_assignment, fx_assignment=None, all_cycles_x=None, all_cycles_y=None, dataset="faust", filename=None):
    plotter = pv.Plotter(shape=(1, 2), off_screen=filename is not None, window_size=[1000, 1000])


    if dataset == "faust":
        degrees_x = 70
        degrees_y = 0
        degrees_z = 120
    else:
        degrees_x = 0
        degrees_y = 0
        degrees_z = 0

    mesh_X = create_pv_mesh(VX, FX)
    mesh_Y = create_pv_mesh(VY, FY)
    mesh_X = rotate_mesh(mesh_X, degrees_x, degrees_y, degrees_z)
    mesh_Y = rotate_mesh(mesh_Y, degrees_x, degrees_y, degrees_z)


    plotter.subplot(0, 0)
    colors = sample_colors(100)
    colors_Y = colors[fy_assignment]
    plotter.add_mesh(mesh_Y, scalars=colors_Y, rgb=True, smooth_shading=False, lighting=True,
                     opacity=1.0)
    plotter.add_title("Mesh Y - source")

    if all_cycles_y is not None:
        for cycle_id, cycle in enumerate(all_cycles_y):
            for row in cycle:
                if row[0] == row[1]:
                    continue
                # line = create_pyvista_cylinder(mesh_Y.points[row[0]], mesh_Y.points[row[1]])
                line = create_pyvista_arrow(mesh_Y.points[row[0]], mesh_Y.points[row[1]], scale=0.02)
                plotter.add_mesh(line, color=colors[cycle_id])


    plotter.subplot(0, 1)
    if fx_assignment is not None:
        colors_X = colors[fx_assignment]
        plotter.add_mesh(mesh_X, scalars=colors_X, rgb=True, smooth_shading=False, lighting=True,
                         opacity=1.0)
    else:
        plotter.add_mesh(mesh_X, color="lightgrey", rgb=False, smooth_shading=False, lighting=True,
                         opacity=1.0)
    plotter.add_title("Mesh X - Target")

    if all_cycles_x is not None:
        for cycle_id, cycle in enumerate(all_cycles_x):
            for row in cycle:
                if row[0] == row[1]:
                    continue
                # line = create_pyvista_cylinder(mesh_X.points[row[0]], mesh_X.points[row[1]])
                line = create_pyvista_arrow(mesh_X.points[row[0]], mesh_X.points[row[1]], scale=0.02)
                plotter.add_mesh(line, color=colors[cycle_id])

    plotter.link_views()


    if filename is None:
        plotter.show()
    else:
        plotter.render()
        plotter.export_html(filename)
        plotter.close()


def plot_dense_matching(VX, FX, VY, FY, p2p, dataset="faust", filename=None):
    plotter = pv.Plotter(shape=(1, 2), off_screen=filename is not None, window_size=[1000, 1000])

    ambient = 0.1
    specular = 0.2
    diffuse = 0.6
    lighting = True
    show_edges = False
    metallic = True
    silhouette = False
    smooth_shading = False

    if dataset == "faust":
        degrees_x = 70
        degrees_y = 0
        degrees_z = 120
    else:
        degrees_x = 0
        degrees_y = 0
        degrees_z = 0

    mesh_X = create_pv_mesh(VX, FX)
    mesh_Y = create_pv_mesh(VY, FY)
    mesh_X = rotate_mesh(mesh_X, degrees_x, degrees_y, degrees_z)
    mesh_Y = rotate_mesh(mesh_Y, degrees_x, degrees_y, degrees_z)


    index_in_x = p2p[:, 1]
    corresponding_index_in_y = p2p[:, 0]

    colors_x = create_colormap(mesh_X.points)
    colors_y = 0 * VY.copy()
    colors_y[corresponding_index_in_y] = colors_x[index_in_x]

    plotter.subplot(0, 0)

    plotter.add_mesh(mesh_X, scalars=colors_x, rgb=True, lighting=lighting, ambient=ambient,
                     silhouette=silhouette,
                     specular=specular, show_edges=show_edges, diffuse=diffuse, metallic=metallic,
                     smooth_shading=smooth_shading)

    plotter.subplot(0, 1)
    plotter.add_mesh(mesh_Y, scalars=colors_y, rgb=True, lighting=lighting, ambient=ambient,
                     silhouette=silhouette,
                     specular=specular, show_edges=show_edges, diffuse=diffuse, metallic=metallic,
                     smooth_shading=smooth_shading)
    plotter.link_views()

    if filename is None:
        plotter.show()
    else:
        plotter.render()
        plotter.export_html(filename)
        plotter.close()





