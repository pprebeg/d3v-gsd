import openmesh as om
import numpy as np
import copy
import sys
import csv
import time
from functools import reduce  # Required in Python 3
import operator
import pygmsh as pg

# contains all universal methods that are inherited by specific mesh class
class DBB_UniversalMesh():

    def update_points(self):
        self.points = self.mesh.points()

    def update_am(self):  # arithmetic middle of points
        self.am = self.points.sum(0) / self.points.shape[0]

    def update_vi(self):
        self.vi = np.arange(self.points.shape[0])

    def update_fvi(self):
        self.fvi = self.mesh.face_vertex_indices()

    def update_evi(self):
        self.evi = self.mesh.edge_vertex_indices()

    def update_vfi(self):
        self.vfi = self.mesh.vertex_face_indices()

    def update_efi(self):
        self.efi = self.mesh.edge_face_indices()

    def update_ffi(self):
        self.ffi = self.mesh.face_face_indices()

    def update_isolated_vi(self):
        self.isolated_vi = np.where((self.vfi == -1).all(-1))[0]

    def update_boundary_ei(self):
        self.boundary_ei = np.where((self.efi == -1).any(-1))[0]

    def update_tp(self):  # triangle points
        self.tp = self.points[self.fvi]

    def update_ep(self):  # edge points
        self.ep = self.points[self.evi]

    def update_elen(self):  # edge lenghts
        self.elen = ((self.ep[:, 1] - self.ep[:, 0]) ** 2).sum(-1) ** 0.5

    def update_avg_edge(self):  # average edge
        self.avg_edge = self.elen.sum() / self.elen.shape[0]

    def update_tmidp(self):
        self.tmidp = self.tp.sum(1) / 3.0

    def update_normals(self):
        normals = np.cross(self.tp[:, 1] - self.tp[:, 0], self.tp[:, 2] - self.tp[:, 0])
        self.normal_len = np.expand_dims(((normals ** 2).sum(-1)) ** 0.5, -1)  # normal lenghts
        # with np.errstate(invalid = "ignore"):		#removes warnings of runtime errors encountering div/0 because form mesh has bad faces in it and it shoudnt affect cutting		-		moved to start of cut_mesh
        self.normals = normals / self.normal_len  # unit vectors

    def update_point_middle(self):
        self.point_middle = self.points.sum(0) / self.points.shape[0]

    def update_all(self):
        self.update_points()
        self.update_am()
        self.update_vi()
        self.update_fvi()
        self.update_evi()
        self.update_vfi()
        self.update_efi()
        self.update_ffi()
        self.update_isolated_vi()
        self.update_boundary_ei()
        self.update_tp()
        self.update_ep()
        self.update_elen()
        self.update_avg_edge()
        self.update_tmidp()
        self.update_normals()
        self.update_point_middle()
        self.calc_mesh_volume_and_centroid()

    def is_mesh_closed(self):
        if self.ffi.shape[
            1] <= 2:  # if all faces are only connected to 2 or less other faces, mesh i not closed by default
            return False
        if (self.ffi == -1).any():
            return False
        else:
            return True

    # regenerates mesh from new self.points and self.fvi, updates other values
    def regenerate_mesh(self, points, fvi):
        self.mesh = om.TriMesh(points, fvi)
        self.update_all()

    # extracts a mesh from existing mesh by its fh idx
    def extract_mesh(self, fh_idx_to_extract,
                     mesh_vh_idx_to_sync=None):  # mesh vh to sync are vh idx for witch we need their new idx
        extracted_fvi = self.fvi[fh_idx_to_extract]
        extracted_vi = np.unique(extracted_fvi.flatten())
        new_points = self.points[extracted_vi]
        new_fvi = arr_replace(extracted_fvi, extracted_vi, np.arange(extracted_vi.shape[0]))
        if mesh_vh_idx_to_sync is None:
            return om.TriMesh(new_points, new_fvi)
        else:
            synced_vh = np.intersect1d(extracted_vi, mesh_vh_idx_to_sync, return_indices=True)[
                1]  # intersect1d returns 1, common elements, 2,indices of first occurace in arr1 3, first comm in arr2 ; new vh idx is equal to indices of occurance in first array
            return (om.TriMesh(new_points, new_fvi), synced_vh)

    def flip_mesh_face_orientation(self):
        flipped_fvi = np.flip(self.fvi, axis=1)
        self.mesh = om.TriMesh(self.points, flipped_fvi)
        self.update_all()

    # removes degenerate faces, isolated vertices and coincident nodes
    def clean_mesh(self, vh_idx_to_sync=None):
        # remove degenerate faces
        bool = (np.isclose(self.normal_len, 0.0)).flatten()
        good_fvi = np.delete(self.fvi, bool, 0)
        self.regenerate_mesh(self.points, good_fvi)
        # remove isolated vertices
        if self.isolated_vi.shape[0] > 0:  # if there are isolated vi: delete them
            synced_vh = self.delete_isolated_vertices(
                vh_idx_to_sync)  # if vh_idx_to_sync was entered as none, this value is also none

        if isinstance(vh_idx_to_sync, np.ndarray):
            data = hard_merge_meshes([self.mesh, ], [vh_idx_to_sync, ])
            self.mesh = data[0]
            self.update_all()
            return data[1]  # return synced vh
        else:
            self.mesh = hard_merge_meshes([self.mesh, ])
            self.update_all()
            return None

    # make sure that isolated vertices are not in vh_idx_to_sync, if they are in, resulting synced array will have -1 in their place
    def delete_isolated_vertices(self, vh_idx_to_sync=None):
        # delete isolated vertices:
        clean_points = np.delete(self.points, self.isolated_vi, 0)
        # sync fvi:
        good_vi = np.delete(self.vi, self.isolated_vi, 0)
        new_good_vi_idx = np.arange(good_vi.shape[0])

        if ~np.equal(np.sort(good_vi), new_good_vi_idx).all():  # if fvi have been changed:
            new_fvi = arr_replace(self.fvi, good_vi, new_good_vi_idx)  # replace good vi in fvi with their indices
        else:  # if not, use old
            new_fvi = self.fvi

        if vh_idx_to_sync is None:
            self.regenerate_mesh(clean_points, new_fvi)
            return None
        else:
            # replace isolated_vi in array with -1
            synced_vh_idx = replace(vh_idx_to_sync, self.isolated_vi, -1)
            # replace good vi with their new ids
            synced_vh_idx = replace(vh_idx_to_sync, good_vi, new_good_vi_idx)
            self.regenerate_mesh(clean_points, new_fvi)
            return synced_vh_idx

    def move_mesh(self, move_vector):
        self.mesh = om.TriMesh(self.points + move_vector, self.fvi)
        self.update_all()

    def calc_mesh_volume_and_centroid(self):
        a = self.tp[:, 0]
        b = self.tp[:, 1]
        c = self.tp[:, 2]

        volumes = (a * np.cross(b, c)).sum(-1) / 6.0
        self.volume = volumes.sum()

        centroids = (self.tp.sum(1)) / 4.0
        self.centroid = (centroids * np.expand_dims(volumes, -1)).sum(0) / self.volume


class DBB_Block(DBB_UniversalMesh):
    def __init__(self, block_dims=np.array([20, 6, 3]), block_position=np.array([0, 0, 0])):
        self.mesh = make_block(block_dims, block_position)
        self.update_all()

    def update_block_dims(self):
        self.block_dims = self.points[7] - self.points[0]  # subtract opposite corners for block dims

    def update_block_position(self):
        self.block_position = self.points[0]

    def update_all(self):
        self.update_points()
        self.update_block_dims()
        self.update_block_position()
        self.update_vi()
        self.update_fvi()
        self.update_evi()
        self.update_vfi()
        self.update_efi()
        self.update_ffi()
        self.update_isolated_vi()
        self.update_boundary_ei()
        self.update_tp()
        self.update_tmidp()
        self.update_normals()
        self.update_point_middle()
        self.calc_mesh_volume_and_centroid()

    def get_outside_and_inside_points(self, Form):
        p1 = np.expand_dims(Form.tp, 1)
        direction = np.expand_dims(Form.scaled_tp, 1)
        p2 = np.concatenate((p1[:, :, [0, 1]], direction[:, :, [1]]), 2)
        p3 = np.concatenate((p1[:, :, [1, 2]], direction[:, :, [2]]), 2)
        p4 = np.concatenate((p1[:, :, [2, 0]], direction[:, :, [0]]), 2)
        p1 = p1[:, :, ::-1]  # reverse so normal points down
        cut_tet_planes = np.concatenate((p1, p2, p3, p4), 1).reshape(-1, 3, 3)
        dist = dist_p_plane(self.points, cut_tet_planes)

        dist = np.array_split(dist, cut_tet_planes.shape[0] / 4,
                              axis=1)  # dist p plane ne daje dobre dimenzije pa ih split i stack poreda kako treba
        dist = np.stack(dist, 1)
        bool = np.where((dist <= 0.0).all(-1))

        self.outside_vi = np.unique(bool[0])
        self.inside_vi = np.delete(self.vi, self.outside_vi, 0)

    def delete_outside_points(self):
        bad_fh = self.vfi[self.outside_vi]
        new_points = np.delete(self.points, self.outside_vi, 0)
        new_fvi = np.delete(self.fvi, bad_fh, 0)
        print(new_points, new_fvi)
        self.regenerate_mesh(new_points, new_fvi)


class DBB_Form(DBB_UniversalMesh):

    def __init__(self, mesh):  # form mesh is imported from file
        self.scale = 1.001  # for scaling in out/in vertex identification
        self.mesh = mesh
        self.update_all()

    def update_scaled_points(self):
        self.scaled_points = (self.points - self.am) * self.scale  # + form_am*scale_move
        self.scaled_points = self.scaled_points + self.am

    def update_scaled_tp(self):
        self.scaled_tp = self.scaled_points[self.fvi]

    def update_all(self):  # new update because of scaling:
        self.update_points()
        self.update_am()
        self.update_vi()
        self.update_fvi()
        self.update_evi()
        self.update_vfi()
        self.update_efi()
        self.update_ffi()
        self.update_isolated_vi()
        self.update_boundary_ei()
        self.update_tp()
        self.update_ep()
        self.update_elen()
        self.update_avg_edge()
        self.update_tmidp()
        self.update_normals()
        self.update_point_middle()
        self.calc_mesh_volume_and_centroid()
        self.update_scaled_points()
        self.update_scaled_tp()

    def get_faces_near_block(self, Block, Block_expansion_coefficient=1.5,
                             n=2):  # n is how many times we search for neighbouring faces

        # print(form_mesh_avg_edge)
        min_dims = Block.block_position
        max_dims = min_dims + Block.block_dims

        # get vh inside unexpanded block (form segment inside vh)
        bool = (((min_dims[0] <= self.points[:, 0]) & (self.points[:, 0] <= max_dims[0])) & (
                    (min_dims[1] <= self.points[:, 1]) & (self.points[:, 1] <= max_dims[1])) & (
                            (min_dims[2] <= self.points[:, 2]) & (self.points[:, 2] <= max_dims[2])))
        inside_unexpanded_block_vh = np.where(bool)[0]  # [0] because return tuple

        # get vh inside expanded block (security measure for extracting form segment)
        avg_edge = np.full((1, 3), self.avg_edge)
        expansion_vector = np.concatenate([np.expand_dims(Block.block_dims * Block_expansion_coefficient, 0), avg_edge],
                                          axis=0).max(
            0) / 2  # max value between block dims * exp coefficient and average edge for each dimension, /2 because it will be added on both sides

        min_dims = min_dims - expansion_vector
        max_dims = max_dims + expansion_vector

        bool = (((min_dims[0] <= self.points[:, 0]) & (self.points[:, 0] <= max_dims[0])) & (
                    (min_dims[1] <= self.points[:, 1]) & (self.points[:, 1] <= max_dims[1])) & (
                            (min_dims[2] <= self.points[:, 2]) & (self.points[:, 2] <= max_dims[2])))
        inside_expanded_block_vh = np.where(bool)[0]  # [0] because return tuple

        # get fh that belong to vh inside expanded block:
        inside_expanded_block_fh = self.vfi[inside_expanded_block_vh].flatten()
        inside_expanded_block_fh = np.unique(np.delete(inside_expanded_block_fh, np.where(
            inside_expanded_block_fh == -1)))  # clean -1 and remove duplicates

        # if no inside vh or fh : try full intersection with block and form

        if inside_expanded_block_fh.shape[
            0] == 0:  # aditional check incase there are no form points inside (there is intersection with form but get_faces_near_block2 dosen't catch it)

            data = tri_tri_inter(Block.tp, self.tp)
            inside_expanded_block_fh = np.unique(data[1][1])

            if inside_expanded_block_fh.shape[0] == 0:
                return inside_expanded_block_fh  # return empty array

        # get neighbouring faces:
        near_fh = copy.copy(inside_expanded_block_fh)  # check if same variable later

        for x in range(n):
            near_fh = np.unique(np.append(near_fh, self.ffi[near_fh].flatten()))
            near_fh = np.delete(near_fh, np.where(
                near_fh == -1))  # remove -1 from fvi(ffi has -1 in it to keep array shape when neighbour face number is less than max )

        return near_fh

    def boolean_difference(self, block_data, algorithm=6):  # om_mesh1 is block, om_mesh2 is form
        self.flip_mesh_face_orientation()
        om_block_mesh = block_data.mesh
        om_form_mesh = self.mesh
        with pg.occ.geometry.Geometry() as geom:
            block_vol = om_to_pg_mesh(geom, om_block_mesh)
            form_vol = om_to_pg_mesh(geom, om_form_mesh)

            geom.boolean_difference(block_vol, form_vol)
            mesh = geom.generate_mesh(dim=2,
                                      algorithm=algorithm)  # algorithm 1: MeshAdapt, 2: Automatic, 3: Initial mesh only, 5: Delaunay, 6: Frontal-Delaunay, 7: BAMG, 8: Frontal-Delaunay for Quads, 9: Packing of Parallelograms, 11: Quasi-structured Quad

            cut_mesh_points = np.array(mesh.points)
            cut_mesh_fvi = np.array(mesh.cells_dict["triangle"])
            om_cut_mesh = om.TriMesh(cut_mesh_points, cut_mesh_fvi)

            cut_block_data = DBB_CutBlock(om_cut_mesh)
            cut_block_data.get_outside_and_inside_points(self)

            return cut_block_data


class DBB_CutBlock(DBB_Block):

    def __init__(self, mesh):
        self.mesh = mesh
        self.update_all()


def make_deck(wline_points, subdivide=False):
    central_points = np.empty((0, 3))
    for point in wline_points:
        if np.isclose(point[1], 0) == False:
            central_point = copy.copy(point)
            central_point[1] = 0
            central_points = np.append(central_points, np.expand_dims(central_point, 0), axis=0)

    deck_points = np.append(wline_points, central_points, axis=0)

    w_max_index = wline_points.shape[0]
    c_max_index = central_points.shape[0]

    # pocetni i zadnji fvi koji ne mogu u petlju
    deck_fvi = np.array([[0, 0 + w_max_index, 1], [deck_points.shape[0] - 1, w_max_index - 1, w_max_index - 2]])

    for interval_index in range(len(central_points - 1)):
        fvi = np.array([[w_max_index, w_max_index + 1, 1], [w_max_index + 1, 2, 1]]) + interval_index
        deck_fvi = np.append(deck_fvi, fvi, axis=0)

    if subdivide == False:
        return om.TriMesh(deck_points, deck_fvi)
    elif subdivide == True:
        return subdivide_mesh([om.TriMesh(deck_points, deck_fvi)])


def make_block(block_dims=np.array([20, 6, 3]), move_vector=np.array([0, 0, 0])):
    mesh = om.TriMesh()
    axes = []
    # stvara 2 tocke na svakoj osi
    for dim in block_dims:
        axes.append(np.linspace(0, dim, 2))

    block_corners = np.asarray(np.meshgrid(*axes)).T.reshape(8, 3)
    block_corners += move_vector

    # shrink block to try to avoid errors in boolean operations
    # block_corners[[4,5,6,7]] += np.array([0,0,-0.00005])			#z+ face is lowered
    # block_corners[[0,1,2,3]] += np.array([0,0,+0.00005])			#z- face is increased

    corner_vertices = []
    for corner in block_corners:
        corner_vertices.append(mesh.add_vertex(corner))

    # x+face
    mesh.add_face(corner_vertices[2], corner_vertices[3], corner_vertices[6])
    mesh.add_face(corner_vertices[3], corner_vertices[7], corner_vertices[6])

    # x-face
    mesh.add_face(corner_vertices[0], corner_vertices[4], corner_vertices[1])
    mesh.add_face(corner_vertices[1], corner_vertices[4], corner_vertices[5])

    # y+face
    mesh.add_face(corner_vertices[3], corner_vertices[1], corner_vertices[5])
    mesh.add_face(corner_vertices[3], corner_vertices[5], corner_vertices[7])

    # y-face
    mesh.add_face(corner_vertices[0], corner_vertices[2], corner_vertices[4])
    mesh.add_face(corner_vertices[2], corner_vertices[6], corner_vertices[4])

    # z+face
    mesh.add_face(corner_vertices[4], corner_vertices[6], corner_vertices[5])
    mesh.add_face(corner_vertices[6], corner_vertices[7], corner_vertices[5])

    # z-face
    mesh.add_face(corner_vertices[2], corner_vertices[0], corner_vertices[1])
    mesh.add_face(corner_vertices[2], corner_vertices[1], corner_vertices[3])

    return mesh


def om_to_pg_mesh(geom, om_mesh):
    om_mesh = hard_merge_meshes([om_mesh, ])
    om_points = om_mesh.points().tolist()
    om_hevi = om_mesh.halfedge_vertex_indices().tolist()
    om_fhei = om_mesh.face_halfedge_indices().tolist()

    pg_points = []
    for point in om_points:
        pg_points.append(geom.add_point(point,
                                        1e10))  # point size is 1e10 to ensure that only triangle points are used for face generation

    pg_lines = []
    for hevi in om_hevi:
        pg_lines.append(geom.add_line(pg_points[hevi[0]], pg_points[hevi[1]])),

    pg_loops = []
    for fhei in om_fhei:
        pg_loops.append(geom.add_curve_loop([pg_lines[fhei[0]], pg_lines[fhei[1]], pg_lines[fhei[2]]]))

    pg_surfaces = []
    for loop in pg_loops:
        pg_surfaces.append(geom.add_plane_surface(loop))

    surf_loop = geom.add_surface_loop(pg_surfaces)
    vol = geom.add_volume(surf_loop)

    return vol


def arr_replace(array, values_to_replace,
                values_to_replace_with):  # replaces values of array with values to replace arg2
    if ~isinstance(values_to_replace_with,
                   np.ndarray):  # if values_to_replace_with is not array(maybe integer or float), will make array suitable for futher calculation filled with that value
        values_to_replace_with = np.full(values_to_replace.shape, values_to_replace_with)

    arr_ndim = len(array.shape)
    to_replace_ndim = len(values_to_replace.shape)
    array_exp = np.expand_dims(array, -to_replace_ndim)
    values_to_replace_exp = np.expand_dims(values_to_replace, 0)

    bool = np.equal(array_exp, values_to_replace_exp)
    for i in range(to_replace_ndim - 1):  # element reduction, reduces point array to single boolean
        bool = bool.all(-1)

    ind = np.asarray(np.where(bool))
    values_to_replace_ind = ind[
        -1]  # indices that point to values values_to_replace and values_to_replace_with (witch value is where)
    ind = tuple(ind[:-1])  # indices that point to bad values in array

    array[ind] = values_to_replace_with[values_to_replace_ind]
    return array


def soft_merge_meshes(meshes,
                      vh_idx_to_sync_list=None):  # meshes je lista sa meshevima, vh_idx_to_sync_list sa lista isog lena ko meshes, svaka sadržava array sa vh_idx koji zelimo syncat
    points = np.empty((0, 3))
    merged_fvi = np.empty((0, 3))

    if vh_idx_to_sync_list is None:
        for mesh in meshes:
            mesh_fvi = mesh.face_vertex_indices()
            if mesh_fvi.size == 0:
                continue
            merged_fvi = np.append(merged_fvi, mesh_fvi + points.shape[0],
                                   axis=0)  # +points.shape[0] je tu da poreda face_vertex_indices sa njihovim indexom u novom arrayu
            points = np.append(points, mesh.points(), axis=0)

        return om.TriMesh(points, merged_fvi)

    else:
        synced_vh_idx = []
        for i in range(len(meshes)):
            mesh = meshes[i]
            mesh_fvi = mesh.face_vertex_indices()
            merged_fvi = np.append(merged_fvi, mesh_fvi + points.shape[0],
                                   axis=0)  # +points.shape[0] je tu da poreda face_vertex_indices sa njihovim indexom u novom arrayu
            synced_vh_idx.append(vh_idx_to_sync_list[i] + points.shape[0])
            points = np.append(points, mesh.points(), axis=0)

        return (om.TriMesh(points, merged_fvi), synced_vh_idx)


# merges coincident vertices!
def hard_merge_meshes(meshes, vh_idx_to_sync_list=None):  # vh_idx_to_sync_list is list with numpy arrays
    if vh_idx_to_sync_list is None:
        merged_mesh = soft_merge_meshes(meshes)
        merged_mesh_fvi = merged_mesh.face_vertex_indices()
        merged_mesh_points = merged_mesh.points()

        bool = np.isclose(np.expand_dims(merged_mesh_points, 0), np.expand_dims(merged_mesh_points, 1)).all(
            -1)  # check for coincident points
        # clean diag and lower triangle matrix
        bool[np.diag_indices(bool.shape[0])] = False
        bool = np.triu(bool)
        ind = np.asarray(np.where(bool))
        # remove duplicates incase 3+ vh idx on same point
        data = np.unique(ind[1], return_index=True)  # [0] unique values, [1] their indices in orig array,
        ind = ind[:, data[1]]
        # delete duplicate points, replace duplicate vh_idx in fvi
        # duplicate vh_idx reduction:
        fvi_ind = np.where(np.expand_dims(merged_mesh_fvi, 0) == ind[1].reshape(-1, 1, 1))
        merged_mesh_fvi[fvi_ind[1:3]] = ind[0][fvi_ind[
            0]]  # slice fvi ind because [0] is indice of what vh_idx the fvi were compared to, 1,2 are actual indices of fvi to be replaced
        # syncing fvi after deleting duplicate points:
        vh_to_delete = np.unique(ind[1])
        vh_to_keep = np.delete(np.arange(merged_mesh_points.shape[0]), vh_to_delete, 0)
        merged_mesh_points = np.delete(merged_mesh_points, vh_to_delete, 0)
        fvi_ind = np.where(np.expand_dims(merged_mesh_fvi, 0) == vh_to_keep.reshape(-1, 1, 1))
        merged_mesh_fvi[fvi_ind[1:3]] = fvi_ind[
            0]  # slice fvi ind because [0] is indice of what vh_idx the fvi were compared to, 1,2 are actual indices of fvi to be replaced

        return om.TriMesh(merged_mesh_points, merged_mesh_fvi)

    else:
        data = soft_merge_meshes(meshes, vh_idx_to_sync_list)
        merged_mesh = data[0]
        vh_idx_to_sync_list = data[1]
        merged_mesh_fvi = merged_mesh.face_vertex_indices()
        merged_mesh_points = merged_mesh.points()

        bool = np.isclose(np.expand_dims(merged_mesh_points, 0), np.expand_dims(merged_mesh_points, 1)).all(-1)
        # clean diag and lower triangle matrix
        bool[np.diag_indices(bool.shape[0])] = False
        bool = np.triu(bool)
        ind = np.asarray(np.where(bool))
        # remove duplicates incase 3+ vh idx on same point
        data = np.unique(ind[1], return_index=True)  # [0] unique values, [1] their indices in orig array,
        ind = ind[:, data[1]]  # ind[0] new vh idx for duplicates, ind[1] old vh idx for duplicates
        # delete duplicate points, replace duplicate vh_idx in fvi
        # duplicate vh_idx reduction:
        fvi_ind = np.where(np.expand_dims(merged_mesh_fvi, 0) == ind[1].reshape(-1, 1, 1))
        merged_mesh_fvi[fvi_ind[1:3]] = ind[0][fvi_ind[
            0]]  # slice fvi ind because [0] is indice of what vh_idx the fvi were compared to, 1,2 are actual indices of fvi to be replaced
        # syncing fvi afrer deleting duplicate points:
        vh_to_delete = np.unique(ind[1])
        vh_to_keep = np.delete(np.arange(merged_mesh_points.shape[0]), vh_to_delete, 0)
        merged_mesh_points = np.delete(merged_mesh_points, vh_to_delete, 0)
        fvi_ind = np.where(np.expand_dims(merged_mesh_fvi, 0) == vh_to_keep.reshape(-1, 1, 1))
        merged_mesh_fvi[fvi_ind[1:3]] = fvi_ind[
            0]  # slice fvi ind because [0] is indice of what vh_idx the fvi were compared to, 1,2 are actual indices of fvi to be replaced

        # sync vh idx:
        synced_vh_idx_list = []
        for vh_idx_to_sync in vh_idx_to_sync_list:
            data = np.intersect1d(vh_idx_to_sync, ind[1], return_indices=True)
            vh_idx_to_sync[data[1]] = ind[0][data[2]]

            # syncing vi after deleting duplicate points:
            data = np.where(np.expand_dims(vh_idx_to_sync, 0) == np.expand_dims(vh_to_keep, 1))
            vh_idx_to_sync[data[1:3]] = data[0]

            synced_vh_idx_list.append(vh_idx_to_sync)
        return (om.TriMesh(merged_mesh_points, merged_mesh_fvi), synced_vh_idx_list)


def tri_tri_inter(tri1, tri2):
    data1 = tri_plane_inter(tri1, tri2)
    inter1 = data1[0]
    indices1 = data1[1].T

    data2 = tri_plane_inter(tri2, tri1)
    inter2 = data2[0]
    indices2 = data2[1].T

    intersecting_indices = (intersect2d(indices1, np.flip(indices2, 1))).T
    s1 = inter1[tuple(intersecting_indices)]
    s2 = inter2[tuple(np.flip(intersecting_indices,
                              0))]  # indices are flipped for data2 because triangle and plane points are reversed in order tri_plane_inter(tri2,tri1)

    # remove segment pairs where points of segment are same:
    bad_i = (np.isclose(s1[:, 1], s1[:, 0]).all(-1)) | (np.isclose(s2[:, 1], s2[:, 0]).all(-1))
    s1 = np.delete(s1, bad_i, 0)
    s2 = np.delete(s2, bad_i, 0)
    intersecting_indices = np.delete(intersecting_indices, bad_i, 1)

    data = segment_inter(s1, s2)  # (points, segments witch had valid intersections)
    indices = intersecting_indices[:, data[1]]
    # unify numerically close points
    data3 = unique_close(data[0], return_inverse=True, axis=0)
    point_segments = data3[0][data3[1]]

    return (point_segments, indices)


def unique_close(ar, return_index=False, return_inverse=False, return_counts=False,
                 axis=None):  # a is nxm		#for axis >1 use only when all instances have a duplicate; else they won't be deleted to conserve array shape
    original_shape = ar.shape
    # ensure if axis is None that array is flattened:
    if axis is None:
        axis = len(original_shape) - 1  # last axis as to flatten ar
    # axis must be viable:
    if axis >= len(original_shape):
        raise CustomError("Axis index " + str(axis) + " out of bounds for inputted array with maximum " + str(
            len(original_shape) - 1) + " axis index.")

    original_shape = np.array(original_shape)
    # cuts original shape to flatten it; ex [6,2,3] for axis = 1 turns to [12,3]:
    flattened_shape = np.copy(original_shape[axis:])
    flattened_shape[0] = original_shape[:(axis + 1)].prod()
    flattened_ar = ar.reshape(flattened_shape)

    axes_to_perform_all_on = tuple(range(-len(original_shape) + 1 + axis,
                                         0))  # tuple with -1,-2,-3 in order to make isclose bool a n X n array and compare all numbers in element
    isclose_bool = np.isclose(np.expand_dims(flattened_ar, 0), np.expand_dims(flattened_ar, 1)).all(
        axes_to_perform_all_on)  # makes bool array ex: flat arr = 8x(elem shape), bool = 8x8x(elem shape), then .all() moves it to 8x8
    isclose_bool = np.tril(
        isclose_bool)  # remove upper tri duplicates		#if triu the last instance of duplicate array will be removed

    counts = (isclose_bool.sum(1) - 1)  # -1 to remove counts from diagonal which is always True

    unique_indices = np.where(~counts.astype(bool))[0]
    unique_elements = flattened_ar[unique_indices]
    # sort element indices(elements are actually sorted before return)

    if return_index == False and return_inverse == False and return_counts == False:
        return unique_elements
    else:
        return_values = [unique_elements]
        if return_index:
            return_values.append(unique_indices)
        if return_inverse:
            return_values.append(np.where(isclose_bool[:, unique_indices])[1])
        if return_counts:
            return_values.append(isclose_bool.sum(0)[
                                     unique_indices])  # counts recalced because summing with axis 1 return max count at last occurance of value, instead of first, whereas unique function demands this behaviour

        return return_values


def dist_p_plane(p, pp, pair_mode=False):  # p = nt x 3, pp = np x 3 x 3
    v0 = pp[:, 0]
    u = pp[:, 1] - v0
    v = pp[:, 2] - v0

    n = np.cross(u, v)
    # with np.errstate(invalid = "ignore"):		#removes warnings of runtime errors encountering nan-s witch works as intended  ---- moved to start of cut_mesh
    n = n / ((n ** 2).sum(-1) ** 0.5).reshape(-1, 1)  # jedinicni vektor
    if pair_mode == False:
        w = np.expand_dims(p, 1) - np.expand_dims(v0, 0)
        nv = np.expand_dims(n, 0)

    elif pair_mode == True:
        if p.shape[0] != pp.shape[0]:  # if pairs do not have matching shapes
            raise CustomError("For pair mode triangle points array and points array must have same shape[0]!")
        w = p - v0
        nv = np.expand_dims(n, 0)

    return (w * nv).sum(-1)


def line_plane_inter3(tp,
                      pp):  # assumption is that tp and pp are in pairs with regards to index 0 and they have intersections
    ro = tp
    # shift array index by 1 So ray direction can be subtrated p[1]-p[0] etc.
    a = np.expand_dims(tp[:, 1], 1)
    b = np.expand_dims(tp[:, 2], 1)
    c = np.expand_dims(tp[:, 0], 1)
    s = np.concatenate((a, b, c), axis=1)
    rd = s - ro

    v0 = pp[:, 0, :]
    v0v = np.expand_dims(v0, 1)
    v0v = np.concatenate((v0v, v0v, v0v), axis=1)
    w = v0v - ro
    u = pp[:, 1, :] - v0
    v = pp[:, 2, :] - v0
    n = np.expand_dims(np.cross(u, v), 1)
    n = np.concatenate((n, n, n), axis=1)
    dot = (n * w).sum(2)
    den = (n * rd).sum(2)
    den = np.where(den != 0, den, np.nan)  # replace 0 with nan
    t = np.expand_dims((dot / den), 2)
    with np.errstate(invalid="ignore"):  # removes warnings of runtime errors encountering nan-s witch works as intended
        t[t < 0.0] = np.nan  # replace bad t-s with nan-s
        t[t > 1.0] = np.nan
    pi = ro + rd * t
    # intersection point cleaning:
    sets = np.full((pi.shape[0], 2, 3), 0.0)
    original_indices = set(range(pi.shape[0]))
    bool = np.isnan(pi)
    nan_i = bool.all(2).any(1)  # indices where set has nan
    dup_i = ~nan_i
    nan_pairs = pi[nan_i]
    i = ~np.isnan(nan_pairs).all(2)
    sets[nan_i] = nan_pairs[i].reshape(-1, 2, 3)

    if dup_i.any():
        dup_pairs = pi[dup_i]
        dup_pairs = unique_close(dup_pairs, axis=1).reshape(-1, 2,
                                                            3)  # ako nije dobro; stavi unique_close(duplikati se desavaju kada je t = 0 i t=1 na 2 raya u istom verteksu trokuta; ako nece biti dobro napravi zasebni uvijet pretrazivanja u tri_tri_inter)
        sets[dup_i] = dup_pairs

    # check if all pairs were intersected
    nanset = set(np.where(nan_i)[0])
    dupset = set(np.where(dup_i)[0])

    if len(nanset) + len(dupset) == len(original_indices):
        return sets
    else:
        error_index = original_indices - (nanset + dupset)
        raise CustomError("No intersection for pair at index: " + str(error_index))


def intersect2d(a, b):
    tmp = np.prod(np.swapaxes(a[:, :, None], 1, 2) == b, axis=2)
    i = np.sum(np.cumsum(tmp, axis=0) * tmp == 1, axis=1).astype(bool)
    return a[i]


def segment_inter(s1, s2):  # segments are paralel and colinear, match eachother by 0 axis

    line_vectors = s1[:, 1] - s1[:, 0]
    i = np.where(~np.isclose(line_vectors, 0))  # find indices of where line vector is not 0
    data = np.unique(i[0], return_index=True)  # only 1 point needed

    unique_axis_0 = np.expand_dims(data[0], 0)
    return_index = data[1]
    unique_axis_1 = np.expand_dims(i[1][return_index], 0)

    unique_i = np.append(unique_axis_0, unique_axis_1, 0)  # indices of first points of set lines that are not 0
    v1 = s1[unique_i[0], :, unique_i[1]]  # v are projected points, never has 2 same points in row
    v2 = s2[unique_i[0], :, unique_i[1]]

    v1min = np.min(v1, 1)
    # v1min_bool = v1 == v1min.reshape(-1,1)

    v1max = np.max(v1, 1)
    # v1max_bool = v1 == v1max.reshape(-1,1)

    v2min = np.min(v2, 1)
    # v2min_bool = v2 == v2min.reshape(-1,1)

    v2max = np.max(v2, 1)
    # v2max_bool = v2 == v2max.reshape(-1,1)

    # segments intersect if min or max of one segment is inbetween of, second check is if both points are in segments , third is to avoid segments intersectiong in only 1 point
    intersecting_segment_bool = (((v1min <= v2min) & (v2min <= v1max)) | ((v1min <= v2max) & (v2max <= v1max)) | (
                (v2min <= v1min) & (v1min <= v2max)) | ((v2min <= v1max) & (v1max <= v2max))) & (
                                            (v1min != v2max) & (v2min != v1max))

    # cleaned segments:
    if ~intersecting_segment_bool.all():  # if any segment index is false: make new s and v that intersect (happens when triangle planes intersect, but triangles dont)
        s1 = s1[intersecting_segment_bool]
        s2 = s2[intersecting_segment_bool]
        v1 = v1[intersecting_segment_bool]
        v2 = v2[intersecting_segment_bool]

    v = np.append(v1, v2, 1)  # projection unity points
    s = np.append(s1, s2, 1)

    vmin = np.min(v, 1)  # arreyevi sa min i max vrijednostima
    vmax = np.max(v, 1)
    vmin_bool = (v == vmin.reshape(-1, 1))
    vmax_bool = (v == vmax.reshape(-1, 1))
    vmin_row_where_2 = np.where(vmin_bool.sum(1) == 2)[0]  # where are 2 True instances
    vmax_row_where_2 = np.where(vmax_bool.sum(1) == 2)[0]

    if vmin_row_where_2.shape[0] > 0:  # if there are rows with duplicate True values
        bad_min_i = np.argmax(vmin_bool[vmin_row_where_2],
                              1)  # argmax return first max value; in bool True is 1 ,[0] to remove from list
        vmin_bool[vmin_row_where_2, bad_min_i] = False  # changes the extra True to False

    if vmax_row_where_2.shape[0] > 0:  # if there are rows with duplicate True values
        bad_max_i = np.argmax(vmax_bool[vmax_row_where_2], 1)
        vmax_bool[vmax_row_where_2, bad_max_i] = False

    v_bool = vmin_bool + vmax_bool

    segments = s[~v_bool].reshape(-1, 2, 3)  # remove ~ to return the union interval of segments, instead of difference
    return [segments, intersecting_segment_bool]


def dist_tri_p_plane(tp, pp):  # tp = nt x 3 x 3, pp = np x 3 x 3
    u = pp[:, 1] - pp[:, 0]
    v = pp[:, 2] - pp[:, 0]
    v0 = pp[:, 0]
    n = np.cross(u, v)
    tpv = np.expand_dims(np.swapaxes(tp, 0, 1), 0)  # remake to shape(1 x 3 x nt x 3)
    nv = n.reshape(n.shape[0], 1, 1, 3)
    sum = np.sum(tpv * nv, 3)
    d = -(n * v0).sum(1)
    d = d.reshape(d.shape[0], 1, 1)
    dn = ((n ** 2).sum(1)) ** 0.5
    dn = dn.reshape(dn.shape[0], 1, 1)
    with np.errstate(invalid="ignore"):  # removes warnings of runtime errors encountering div with 0
        return np.swapaxes(((sum + d) / dn), 0, -1)


def tri_plane_inter(tp, pp):  # po starom ili da samo lupim line_plane inter?
    d = dist_tri_p_plane(tp, pp)
    return_arr = np.full((tp.shape[0], pp.shape[0], 2, 3), np.nan)
    Tin_bool = d < 0.0
    Tin = np.where(Tin_bool, 1, 0).sum(1)
    Ton_bool = d == 0
    Ton = np.full(Ton_bool.shape, 0)
    Ton[Ton_bool] = 1
    Ton = Ton.sum(1)
    Tout_bool = d > 0.0
    Tout = np.where(Tout_bool, 1, 0).sum(1)
    count_arr = np.concatenate((Tin.reshape(Tin.shape[0], 1, Tin.shape[1]), Ton.reshape(Ton.shape[0], 1, Ton.shape[1]),
                                Tout.reshape(Tout.shape[0], 1, Tout.shape[1])), axis=1)

    # on points cond = onlen == 2 and (inlen == 1 or outlen == 1), return op_p
    onp_cond = ((count_arr == np.array([[1], [2], [0]])).all(1)) | (
        (count_arr == np.array([[0], [2], [1]])).all(1))  # treba li ovaj cond?
    i_onp = np.where(onp_cond)
    # inter cond = inlen == 2 and outlen == 1 or elif inlen == 1 and outlen == 2   or inlen == 1 and outlen == 1 and onlen == 1, return inter
    inter_cond = ((count_arr == np.array([[2], [0], [1]])).all(1)) | (
        (count_arr == np.array([[1], [0], [2]])).all(1)) | ((count_arr == np.array([[1], [1], [1]])).all(1))
    i_inter = np.where(inter_cond)  # index0 = tp, index1 = pp
    # points and planes with valid intersections:
    intersecting_tp = tp[i_inter[0]]
    intersecting_pp = pp[i_inter[1]]

    inter = line_plane_inter3(intersecting_tp, intersecting_pp)
    # ovo vraca pune trokute
    ontp = tp[i_onp[0]]
    onpp = pp[i_onp[1]]

    where_onp = Ton_bool.transpose(0, 2, 1)[i_onp]
    onp = ontp[np.where(where_onp)].reshape(-1, 2, 3).astype(np.float64)  # setovi tocaka koji su na planeu

    # return_inter = np.append(inter, onp,0)
    return_arr[i_onp] = onp

    return_arr[i_inter] = inter
    indices = np.append(i_inter, i_onp, 1)  # row 0 is triangle index and below is its pair plane index

    return (return_arr, indices)


def subdivide_mesh(mesh_list, c=0, n=1):  # face po face subdividamo n puta,c je counter
    if c < n:
        new_meshes = []
        for mesh in mesh_list:
            mesh_points = mesh.points()
            mesh_fvi = mesh.face_vertex_indices().tolist()
            mesh_hei = mesh.face_halfedge_indices().tolist()  # lista sa 3 vrijednosti unutra
            face_hevi = mesh.halfedge_vertex_indices().tolist()  # heh idx -> vertex indices	#lista [.........] velika sa slistama od 2 point = points[vindices]
            for i in range(len(mesh_fvi)):  # i je idx od fh
                face_points = np.empty((0, 3))
                midpoints = np.empty((0, 3))
                for j in mesh_hei[i]:  # j je idx od heh za halfedgeve na tom faceu /za svaki halfedge handleidx u faceu

                    hevi = (face_hevi[j])
                    halfedge_points = mesh_points[hevi]  # array
                    face_points = np.append(face_points, np.expand_dims(halfedge_points[0], axis=0),
                                            axis=0)  # zadrzavanje orijentacije
                    midpoint = halfedge_points[0] + (halfedge_points[1] - halfedge_points[0]) * 0.5
                    midpoints = np.append(midpoints, np.expand_dims(midpoint, axis=0), axis=0)
                new_mesh = om.TriMesh()
                vhandles = []
                fhandles = []
                for point in np.append(face_points, midpoints, axis=0):
                    vhandles.append(new_mesh.add_vertex(point))

                fhandles.append(new_mesh.add_face(vhandles[0], vhandles[3], vhandles[5]))
                fhandles.append(new_mesh.add_face(vhandles[3], vhandles[1], vhandles[4]))
                fhandles.append(new_mesh.add_face(vhandles[5], vhandles[3], vhandles[4]))
                fhandles.append(new_mesh.add_face(vhandles[5], vhandles[4], vhandles[2]))
                new_meshes.append(new_mesh)

        return subdivide_mesh(new_meshes, c=c + 1, n=n)

    else:
        return hard_merge_meshes(mesh_list)


def move_mesh(mesh, move_vector):
    return om.TriMesh(mesh.points() + move_vector, mesh.face_vertex_indices())


def calc_mesh_volume(mesh):  # for use in dbb.py
    points = mesh.points()
    fvi = mesh.face_vertex_indices()
    tp = points[fvi]
    a = tp[:, 0]
    b = tp[:, 1]
    c = tp[:, 2]

    volumes = (a * np.cross(b, c)).sum(-1) / 6.0
    volume = volumes.sum()
    return volume


def is_mesh_closed(mesh):  # for use in dbb.py
    ffi = mesh.face_face_indices()
    if ffi.shape[1] <= 2:  # if all faces are only connected to 2 or less other faces, mesh i not closed by default
        return False
    if (ffi == -1).any():
        return False
    else:
        return True


def cut_mesh(form_mesh, block_dims=np.array([5, 5, 5]), block_position=np.array([0, 0, 0])):
    with np.errstate(invalid="ignore"):  # supress warnings
        form_data = DBB_Form(form_mesh)
        block_data = DBB_Block(block_dims, block_position)

        # getting form segment:
        try:  # if there is an error in tri_tri inter do pygmsh boolean difference on whole form
            form_segment_fh = form_data.get_faces_near_block(block_data, Block_expansion_coefficient=1.5, n=2)
        except:
            form_data.clean_mesh()
            try:
                cut_block_data = form_data.boolean_difference(block_data)
            except:  # incase there is no intersection
                return block_data.mesh
            else:
                if cut_block_data.is_mesh_closed() == False:
                    print("Mesh was not closed, switching meshing algorithm to BAMG.")
                    try:
                        cut_block_data = form_data.boolean_difference(block_data, 7)
                    except:  # if it couldn't be meshed with BAMG return block mesh:
                        print("Meshing with BAMG failed, returning uncut block.")
                        return block_data.mesh

                    print("Error in intersection; calculating cut block from full form:")
                return cut_block_data.mesh

        else:  # if there are no errors continue normally:
            block_data.get_outside_and_inside_points(form_data)
            # if there are no form segment faces or if there are no inside points block does not intersect with form
            if (form_segment_fh.shape[0] == 0):
                return block_data.mesh
            form_segment_mesh = form_data.extract_mesh(form_segment_fh)
            form_segment_data = DBB_Form(form_segment_mesh)
            form_segment_data.clean_mesh()  # form segment is not properly connected on symmetric plane and boundary edges can cause problems for algorithm, so we clean the mesh and merge those edges

            try:
                cut_block_data = form_segment_data.boolean_difference(block_data)
            except:  # incase there is no intersection
                return block_data.mesh
            else:
                if cut_block_data.is_mesh_closed() == False:
                    print("Mesh was not closed, switching meshing algorithm to BAMG.")
                    form_segment_data.flip_mesh_face_orientation()  # was flipped once in first bool difference
                    try:
                        cut_block_data = form_segment_data.boolean_difference(block_data, 7)
                    except:  # if it couldn't be meshed with BAMG return block mesh:
                        print("Meshing with BAMG failed, returning uncut block.")
                        return block_data.mesh

                return cut_block_data.mesh

