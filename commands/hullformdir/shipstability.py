from hullformdir.hullform import *
from scipy import optimize
import numpy as np
import time

class ShipStability():
    def __init__(self, hull_form:HullForm,sea_density = 1.025):
        self._hf = hull_form
        self._xmf=self._hf.get_x_main_frame_from_mesh()
        self._sea_density = sea_density
        self._ship_weight:float =0.0
        self._ship_CG:np.ndarray = np.zeros(3)

    def calculate_displacement_and_displacementCG(self):
        hwaterline = 5.0  # srednja visina vodne linije
        plane_point = np.array([0, 0, hwaterline])
        plane_normal = np.array([0, 0, -1])

        fvs = self._hf.mesh.fv_indices().tolist()
        points = self._hf.mesh.points().tolist()
        # try:
        new_fvs, new_pts = self.get_mesh_below_inclined_waterline(fvs, points, plane_point, plane_normal)
        displacement, displacementCG = self.calculate_displacement_and_centroid(new_fvs, new_pts)
        print('displacement',displacement)
        print('displacement CG', displacementCG)
        testgeo = GeometryExtension('Displ_Calc_Mesh')
        testgeo.mesh = om.TriMesh(new_pts,new_fvs)
        testgeo.emit_geometry_built()

    def calculate_drought_horizontal_waterline(self):
        self._ship_weight: float = 2000.0
        Dmid,Dmax = self._hf.get_z_mid_z_max_from_mesh()

        res = optimize.root(self.calculate_diference_ship_weight_displacement, Dmid)
        print(res)
        x=res['x']
        T=x[0]
        print('T',T)

    def calculate_diference_ship_weight_displacement(self, T:float):
        diff = self._ship_weight - self.calculate_displacement_horizontal_waterline(T)
        return diff

    def calculate_displacement_horizontal_waterline(self, T:float):
        plane_point = np.array([0, 0, T])
        plane_normal = np.array([0, 0, -1])
        fvs = self._hf.mesh.fv_indices().tolist()
        points = self._hf.mesh.points().tolist()
        # try:
        new_fvs, new_pts = self.get_mesh_below_inclined_waterline(fvs, points, plane_point, plane_normal)
        displacement, centroid = self.calculate_displacement_and_centroid(new_fvs, new_pts)
        return displacement

    def test(self):
        fvs = self.mesh.fv_indices().tolist()
        points = self.mesh.points().tolist()

        hwaterline = 5.0  # srednja visina vodne linije
        plane_point = np.array([0, 0, hwaterline])
        plane_normal = np.array([0, 0, -1])

        # try:
        new_fvs, new_pts = self.get_mesh_below_inclined_waterline(fvs, points, plane_point, plane_normal)
        displacement, centroid = self.calculate_displacement_and_centroid(new_fvs, new_pts)
        # except BaseException as e:
        #    print(e)
        # except:
        #    pass

        fvs2calcWl = new_fvs
        points2calcWl = new_pts
        xmf = self.shipdata["loa_val"] / 2

        bcwl = self.getBasicDataUsingTrianglesProjectedToWaterline(hwaterline, xmf, fvs2calcWl, points2calcWl)
        h = bcwl[0]
        volume = bcwl[1]
        area = bcwl[2]
        Xwl = bcwl[3]
        KBz = bcwl[4]
        KBx = bcwl[5]
        Ib = bcwl[6]
        Il = bcwl[7]

        def root_func(initial_guess):
            return initial_guess - self.calculate_displacement_and_centroid(fvs, points)[0]

        print('LH: bcwl=', bcwl, ' displacement=', displacement, ' centroid=', centroid, )
        print(optimize.root(root_func, 6500))

        mesh2calcWl2 = self.get_tria_for_calculation(fvs, points, hwaterline)
        fvs2calcWl2 = mesh2calcWl2[0]
        points2calcWl2 = mesh2calcWl2[1]

        bcwl2 = self.getBasicDataUsingTrianglesProjectedToWaterline(hwaterline, xmf, fvs2calcWl2, points2calcWl2)
        h = bcwl2[0]
        volume = bcwl2[1]
        area = bcwl2[2]
        Xwl = bcwl2[3]
        KBz = bcwl2[4]
        KBx = bcwl2[5]
        Ib = bcwl2[6]
        Il = bcwl2[7]
        print('LJE:', bcwl2)

        print('LH-LJE:', np.array(bcwl) - np.array(bcwl2))
        return

    # import sympy, Point3D and Plane

    # fvs = [(1, 2, 40), (2, 3, 42), ...] Trokut, lista imena tocaka
    def get_mesh_below_inclined_waterline(self, fvs, points, plane_point, plane_normal):
        def insert_point_based_on_sign(deck_points_pos, deck_points_neg, point):
            if point[1] < 0:
                deck_points_neg.append(point)
            else:
                deck_points_pos.append(point)

        def make_new_triangles(fix_name, points, trianagles_bwl, deck_points):
            n = len(points)
            for i in range(len(deck_points) - 1):
                points.append(deck_points[i])
                points.append(deck_points[i + 1])
                trianagles_bwl.append([fix_name, n, n + 1])
                n += 2

        d = plane_point.dot(plane_normal)
        trianagles_bwl = []
        deck_points_pos = []
        deck_points_neg = []
        for fh in fvs:  # for TROKUT in TROKUTI:
            points_bwl = []
            points_owl = []
            bwl_order = []
            owl_order = []
            i = 0
            for vh in fh:  # for TOCKA in TROKUT
                p_np = np.array(points[vh])
                distance_to_plane = p_np.dot(plane_normal) - d
                if distance_to_plane < 0:
                    points_owl.append(points[vh])
                    owl_order.append(i)
                else:
                    points_bwl.append(points[vh])
                    bwl_order.append(i)
                i += 1
            if len(points_bwl) == 3:
                trianagles_bwl.append(fh)
                continue
            if len(points_owl) == 3:
                continue
            if len(points_bwl) == 1:
                a = np.array(points_bwl[0])  # a je jedina ispod vodne linije
                b = np.array(points_owl[0])
                c = np.array(points_owl[1])

                t1 = (d - plane_normal.dot(a)) / plane_normal.dot(b - a)
                contact1 = a + (b - a) * t1

                t2 = (d - plane_normal.dot(a)) / plane_normal.dot(c - a)
                contact2 = a + (c - a) * t2

                insert_point_based_on_sign(deck_points_pos, deck_points_neg, contact1)
                insert_point_based_on_sign(deck_points_pos, deck_points_neg, contact2)

                print('kontakt točke', contact1, contact2)

                n = len(points)
                if bwl_order[0] == 1:
                    fh_new = np.array([fh[bwl_order[0]], n + 1, n])
                    trianagles_bwl.append(fh_new)
                else:
                    fh_new = np.array([fh[bwl_order[0]], n, n + 1])
                    trianagles_bwl.append(fh_new)

            elif len(points_bwl) == 2:
                a = np.array(points_owl[0])  # a je jedina iznad vodne linije
                b = np.array(points_bwl[0])
                c = np.array(points_bwl[1])

                t1 = (d - plane_normal.dot(a)) / plane_normal.dot(b - a)
                contact1 = a + (b - a) * t1

                t2 = (d - plane_normal.dot(a)) / plane_normal.dot(c - a)
                contact2 = a + (c - a) * t2

                insert_point_based_on_sign(deck_points_pos, deck_points_neg, contact1)
                insert_point_based_on_sign(deck_points_pos, deck_points_neg, contact2)

                print('kontakt točke', contact1, contact2)

                n = len(points)
                if owl_order[0] == 1:
                    fh_new = np.array([n, n + 1, fh[bwl_order[1]]])
                    trianagles_bwl.append(fh_new)
                    fh_new = np.array([n, fh[bwl_order[1]], fh[bwl_order[0]]])
                    trianagles_bwl.append(fh_new)
                else:
                    fh_new = np.array([n, fh[bwl_order[1]], n + 1])
                    trianagles_bwl.append(fh_new)
                    fh_new = np.array([n, fh[bwl_order[0]], fh[bwl_order[1]]])
                    trianagles_bwl.append(fh_new)

        deck_points_pos.sort(key=lambda point: point[0])
        deck_points_neg.sort(key=lambda point: point[0])

        fix = deck_points_pos.pop(0)  # fix point izbacujemo iz liste deck points pos
        points.append(fix)
        fix_name = len(points) - 1  # dodavanje fix pointu ime
        last_neg_point = None

        for point in deck_points_pos:
            if point[0] == deck_points_pos[-1][0] and point[1] == 0:  # provjera koja tocka na krmi je na sredini
                deck_points_neg.append(point)  # fix point dodajemo u listu deck point neg
                last_neg_point = len(deck_points_neg)  # dodavanje srednjoj tocki na krmi ime
                break

        make_new_triangles(fix_name, points, trianagles_bwl, deck_points_pos)
        make_new_triangles(fix_name, points, trianagles_bwl, deck_points_neg)

        second_to_last_neg_point = points[-2]
        second_to_last_neg_point = len(points) - 2
        trianagles_bwl[-1] = [fix_name, second_to_last_neg_point, last_neg_point]
        points.pop()

        return trianagles_bwl, points

    def calculate_displacement_and_centroid(self, fvs, points):
        displacement = 0
        centroid = 0
        for fh in fvs:
            v, t = self.calculate_tethraedar_displacement_and_centroid(fh, points)
            displacement += v
            centroid += v * t
        return displacement, centroid / displacement

    def calculate_tethraedar_displacement_and_centroid(self, fh, points):
        p1 = np.array(points[fh[0]])
        p2 = np.array(points[fh[1]])
        p3 = np.array(points[fh[2]])

        v321 = p3[0] * p2[1] * p1[2]
        v231 = p2[0] * p3[1] * p1[2]
        v312 = p3[0] * p1[1] * p2[2]
        v132 = p1[0] * p3[1] * p2[2]
        v213 = p2[0] * p1[1] * p3[2]
        v123 = p1[0] * p2[1] * p3[2]
        displacement_of_tetrahedron = (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123)
        centroid_of_tetrahedron = (p1 + p2 + p3) / 4

        return displacement_of_tetrahedron, centroid_of_tetrahedron