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

    def calculate_displacement_and_displacementCG_example(self):
        hwaterline = 5.0  # srednja visina vodne linije
        plane_point = np.array([0, 0, hwaterline])
        plane_normal = np.array([0, 0.0, -1])
        displacement, displacementCG, new_fvs, new_pts = self.calculate_displacement_and_displacementCG(plane_point,plane_normal)
        print('displacement', displacement)
        print('displacement CG', displacementCG)
        testgeo = GeometryExtension('Displ_Calc_Mesh')
        testgeo.mesh = om.TriMesh(new_pts, new_fvs)
        testgeo.emit_geometry_built()

    def calculate_displacement_and_displacementCG(self,plane_point,plane_normal):

        fvs = self._hf.mesh.fv_indices().tolist()
        points = self._hf.mesh.points().tolist()
        # try:
        new_fvs, new_pts = self.get_mesh_below_inclined_waterline(fvs, points, plane_point, plane_normal)
        displacement, displacementCG = self.calculate_displacement_and_centroid(new_fvs, new_pts)
        return displacement, displacementCG, new_fvs, new_pts


    def calculate_drought_horizontal_waterline(self):
        self._ship_weight: float = 2000.0
        Dmid,Dmax = self._hf.get_z_mid_z_max_from_mesh()

        res = optimize.root(self.calculate_diference_ship_weight_displacement, Dmid)
        print(res)
        x=res['x']
        T=x[0]
        print('T',T)

    def calculate_trim(self):
        self._ship_weight: float = 3160
        self._ship_CG: float = np.array([49.1,0.0,0.0])
        Dmid,Dmax = self._hf.get_z_mid_z_max_from_mesh()


        res = optimize.root(self.calculate_diference_ship_weight_displacement_CG,np.array([Dmid,0.0]))
        print(res)
        x=res['x']
        T=x[0]
        pl_nv_x=x[1]
        print('T',T)
        print('pl_nv_x', pl_nv_x)

    def calculate_diference_ship_weight_displacement_CG(self, x):
        T = x[0]
        pl_nv_x = x[1]
        plane_point = np.array([0, 0, T])
        plane_normal = np.array([pl_nv_x, 0.0, -1])
        displacement, displacementCG, new_fvs, new_pts = self.calculate_displacement_and_displacementCG(plane_point,
                                                                                                        plane_normal)
        diff1 = self._ship_weight - displacement
        # Racunanje projekcija tezista istisnine i tezista broda na ravninu vodne linije
        displacementCG_proj = self.project_point_on_plane(displacementCG, plane_point, plane_normal)
        shipCG_proj = self.project_point_on_plane(self._ship_CG, plane_point, plane_normal)
        # Racunanje razlike izmedu projekcija tezista istisnine i tezista broda
        diff2 = np.linalg.norm(displacementCG_proj - shipCG_proj)
        diff = diff1 ** 2 + diff2 ** 2
        #return diff, 0
        return diff1,diff2

    def project_point_on_plane(self, point, plane_point, plane_normal):
        d = -plane_normal.dot(plane_point)
        t = -(plane_normal.dot(point) + d) / (plane_normal.dot(plane_normal))
        projection = point + t * plane_normal
        return projection

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
        # Add central point for hull_cover mesh
        cp = np.zeros(3)
        icp = len(points)
        points.append(cp)

        d = plane_point.dot(plane_normal)
        trianagles_bwl = []
        i_wl_0 = len(points)
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

                n = len(points)
                points.append(contact1)
                points.append(contact2)

                # add hull triangle
                if bwl_order[0] == 1:
                    fh_new = np.array([fh[bwl_order[0]], n + 1, n])
                    trianagles_bwl.append(fh_new)
                    # add hull_cover triangle
                    fh_new = np.array([n, n + 1, icp])
                    trianagles_bwl.append(fh_new)
                else:
                    fh_new = np.array([fh[bwl_order[0]], n, n + 1])
                    trianagles_bwl.append(fh_new)
                    # add hull_cover triangle
                    fh_new = np.array([n+1, n, icp])
                    trianagles_bwl.append(fh_new)

            elif len(points_bwl) == 2:
                a = np.array(points_owl[0])  # a je jedina iznad vodne linije
                b = np.array(points_bwl[0])
                c = np.array(points_bwl[1])

                t1 = (d - plane_normal.dot(a)) / plane_normal.dot(b - a)
                contact1 = a + (b - a) * t1

                t2 = (d - plane_normal.dot(a)) / plane_normal.dot(c - a)
                contact2 = a + (c - a) * t2

                n = len(points)
                points.append(contact1)
                points.append(contact2)
                # add hull triangles
                if owl_order[0] == 1:
                    fh_new = np.array([n, n + 1, fh[bwl_order[1]]])
                    trianagles_bwl.append(fh_new)
                    fh_new = np.array([n, fh[bwl_order[1]], fh[bwl_order[0]]])
                    trianagles_bwl.append(fh_new)
                    # add hull_cover triangle
                    fh_new = np.array([n+1, n, icp])
                    trianagles_bwl.append(fh_new)
                else:
                    fh_new = np.array([n, fh[bwl_order[1]], n + 1])
                    trianagles_bwl.append(fh_new)
                    fh_new = np.array([n, fh[bwl_order[0]], fh[bwl_order[1]]])
                    trianagles_bwl.append(fh_new)
                    # add hull_cover triangle
                    fh_new = np.array([n, n + 1, icp])
                    trianagles_bwl.append(fh_new)

        deck_points = np.array(points[i_wl_0:]).tolist()
        xmin = deck_points[0][0]
        xmax = deck_points[0][0]
        ixmax = 0
        ixmin = 0
        for i in range(len(deck_points)):
            if deck_points[i][0] > xmax:
                xmax = deck_points[i][0]
                ixmax = i
            if deck_points[i][0] < xmin:
                xmin = deck_points[i][0]
                ixmin = i
        xcp = (xmax + xmin) / 2.0
        zcp = (deck_points[ixmax][2] + deck_points[ixmin][2]) / 2.0
        cp[0]=xcp
        cp[2]=zcp
        return trianagles_bwl, points

    def generate_hull_cover_triangles(self,points:list,i_wl_0,fvi):
        #Method is not used
        deck_points_ixs=np.arange(i_wl_0,len(points)).tolist()
        deck_points=np.array(points[i_wl_0:]).tolist()
        xmin=deck_points[0][0]
        xmax = deck_points[0][0]
        ixmax=0
        ixmin=0
        for i in range(len(deck_points)):
            if deck_points[i][0]>xmax:
                xmax = deck_points[i][0]
                ixmax =i
            if deck_points[i][0] < xmin:
                xmin = deck_points[i][0]
                ixmin = i
        for i in range(len(deck_points)):
            deck_points[i][0]=deck_points[i][0]-(xmax - xmin)/2
        #circular sort
        deck_points_ixs_s =  [x for _, x in sorted(zip(deck_points, deck_points_ixs), key=lambda c:np.arctan2(c[0][0], c[0][1]))]
        xcp=(xmax+xmin)/2.0
        ycp=(deck_points[ixmax][1]+deck_points[ixmin][1])/2.0
        zcp = (deck_points[ixmax][2] + deck_points[ixmin][2]) / 2.0
        cp=np.array([xcp,ycp,zcp])
        icp=len(points)
        points.append(cp)
        for i in range(len(deck_points_ixs_s)-1):
            fvi.append(np.array([deck_points_ixs_s[i], deck_points_ixs_s[i+1],icp]))
        fvi.append(np.array([deck_points_ixs_s[-1], deck_points_ixs_s[0], icp]))

    def calculate_displacement_and_centroid(self, fvs, points):
        displacement = 0
        centroid = 0
        for fh in fvs:
            v, t = self.calculate_tethraedar_displacement_and_centroid(fh, points)
            displacement += v
            centroid += v * t
        return displacement, centroid / displacement

    def calculate_tethraedar_displacement_and_centroid(self, fh, points):
        try:
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
        except BaseException as error:
            print('An exception occurred: {}'.format(error))
        except:
            print('Unknown exception occurred during signals connection')

        return displacement_of_tetrahedron, centroid_of_tetrahedron