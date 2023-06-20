from hullformdir.hullform import *
from scipy import optimize
import numpy as np
from scipy.spatial.transform import Rotation
import math
import time
import numpy.linalg as la
from  typing import List

def angle_between_vectors_deg(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'
    Return angle range is between 0 and 180° """

    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    ang_rad = np.arctan2(np.clip(sinang,-1.0,1.0), np.clip(cosang,-1.0,1.0))
    return np.rad2deg(ang_rad)

def deg_to_vec(deg):
    return np.cos(np.deg2rad(deg))

class LoadCondition():
    def __init__(self,weight:float= 0.0 ,CG:np.ndarray = np.zeros(3)):
        self._ship_mass: float = weight
        self._ship_CG: np.ndarray = CG

    @property
    def mass(self):
        return self._ship_mass
    @property
    def CG(self):
        return self._ship_CG

class Waterline():
    @classmethod
    def ref_plane_normal(cls):
        return np.array([0.0, 0.0, -1])

    @classmethod
    def get_point_on_plane_close_to_input_point(cls,plane_point:np.ndarray,plane_normal:np.ndarray,in_point:np.ndarray):
        w = np.array([-1.0, 0.0, 0.0])
        v = np.cross(w, plane_normal)
        u = np.cross(plane_normal, v)
        p1 = v*(in_point[1]-plane_point[1])
        p2 = u*(in_point[0]-plane_point[0])
        cp = plane_point+p1+p2
        return cp

    def __init__(self, pp_z:float, pp_x:float=0.0):
        self._plane_point = np.array([pp_x, 0.0, pp_z])
        self._plane_normal =  Waterline.ref_plane_normal()

    def set_plane_normal_component(self,value:float,index:int):
        self._plane_normal[index]= value

    def set_plane_normal_x(self,value:float):
        self.set_plane_normal_component(value,0)

    def set_plane_normal_y(self,value:float):
        self.set_plane_normal_component(value,1)

    def set_plane_normal_z(self,value:float):
        self.set_plane_normal_component(value,2)

    def set_plane_point_component(self, value: float, index: int):
        self._plane_point[index] = value

    def set_plane_point_x(self, value: float):
        self.set_plane_point_component(value,0)

    def set_plane_point_y(self, value: float):
        self.set_plane_point_component(value,1)

    def set_plane_point_z(self, value: float):
        self.set_plane_point_component(value,2)

    def reset_to_heel_angle_deg(self,heel:float):
        self._plane_normal = Waterline.ref_plane_normal()
        # heel is rotation over x axix (y-z plane)
        r = Rotation.from_euler('x', heel, degrees=True)
        self._plane_normal = r.apply(self._plane_normal)

    def get_angle_between_plane_normal_and_ref_plane_normal(self):
        return angle_between_vectors_deg(Waterline.ref_plane_normal(),self._plane_normal)

    def get_trim_angle(self):
        trim_vec = Waterline.ref_plane_normal()
        trim_vec[0]= self.normal[0]
        return angle_between_vectors_deg(Waterline.ref_plane_normal(),trim_vec)

    def set_trim_angle(self,trim):
        self._plane_normal = Waterline.ref_plane_normal()
        r = Rotation.from_euler('y', trim, degrees=True)
        self._plane_normal = r.apply(self._plane_normal)

    def get_heel_angle(self):
        heel_vec= self.normal.copy()
        heel_vec[0] = 0.0
        return angle_between_vectors_deg(Waterline.ref_plane_normal(),heel_vec)


    def get_max_distance_to_mesh(self, fvs, pts):
        max_dist = -np.inf
        most_dist_point = None
        for triangle in fvs:
            for i_p in triangle:
                p = np.array(pts[i_p])
                v = p - self.point  # vektor koji spaja točku na ravnini i točku u trokutu
                distance = np.dot(v, self.normal)  # Udaljenost točke od ravnine
                if distance > max_dist:
                    max_dist = distance
                    most_dist_point = p

        return max_dist, most_dist_point

    def create_and_emmit_geometry(self,name,length:float, width:float):
        w = np.array([-1.0, 0.0, 0.0])
        v = np.cross(w, self.normal)
        u = np.cross(self.normal, v)
        cp = self.point
        points = []
        dl = length*0.6
        dw = width*0.6
        points.append(cp + u * dl + v * dw)
        points.append(cp + u * dl - v * dw)
        points.append(cp - u * dl + v * dw)
        points.append(cp - u * dl - v * dw)
        fvs = []
        fvs.append([0, 1, 3])
        fvs.append([0, 3, 2])
        name+=' point {0}, normal {1}'.format(self.point, self.normal)
        g = GeometryExtension(name)
        g.mesh = om.TriMesh(points,fvs)
        g.emit_geometry_built()



    @property
    def point(self):
        return self._plane_point

    @property
    def normal(self):
        return self._plane_normal



class ShipStability():
    def __init__(self, hull_form:HullForm,main_deck_z,sea_density = 1025.9):
        self._hf = hull_form
        self._xmf = 0.0
        if self._hf is not None:
            self._xmf = self._hf.get_x_main_frame_from_mesh()
        self._sea_density = sea_density
        self._ship_weight:float =0.0
        self._ship_CG:np.ndarray = np.zeros(3)
        self._wl:Waterline = Waterline(0.0,self.xmf)
        self._main_deck_z = main_deck_z
        self._mesh = self.get_mesh_form_closed_with_main_deck()

    def get_mesh_form_closed_with_main_deck(self):
        if self._hf is not None:
            fvs = self._hf.mesh.fv_indices().tolist()
            points = self._hf.mesh.points().tolist()
            plane_point = np.array([self._xmf,0.0,self._main_deck_z])
            plane_normal = Waterline.ref_plane_normal()
            new_fvs, new_pts = self.get_mesh_below_inclined_waterline(fvs, points, plane_point, plane_normal)
            mesh =  om.TriMesh(new_pts,new_fvs)
            testgeo = GeometryExtension('Closed Hull Form')
            testgeo.mesh = om.TriMesh(new_pts, new_fvs)
            testgeo.emit_geometry_built()
            return mesh
        return None
    @property
    def wl(self)->Waterline:
        return self._wl

    def get_drought_and_most_dist_point(self):
        fvs = self._hf.mesh.fv_indices().tolist()
        pts = self._hf.mesh.points().tolist()
        T,p = self._wl.get_max_distance_to_mesh(fvs,pts)
        return T,p


    def calculate_displacement_and_displacementCG_example(self):

        displacement, displacementCG, new_fvs, new_pts = self.calculate_displacement_and_displacementCG()
        print('displacement, m3', displacement)
        print('displacement, t', displacement*self._sea_density/1000.0)
        print('displacement CG', displacementCG)
        testgeo = GeometryExtension('Displ_Calc_Mesh')
        testgeo.mesh = om.TriMesh(new_pts, new_fvs)
        testgeo.emit_geometry_built()


    def calculate_displacement_and_displacementCG(self):
        fvs = self._mesh.fv_indices().tolist()
        points = self._mesh.points().tolist()
        plane_point = self._wl.point
        plane_normal= self._wl.normal
        new_fvs, new_pts = self.get_mesh_below_inclined_waterline(fvs,points,plane_point,plane_normal)
        displacement, displacementCG = self.calculate_displacement_and_centroid(new_fvs, new_pts)
        return displacement, displacementCG, new_fvs, new_pts


    def determine_horizontal_waterline_for_current_weight(self):
        z_mid,z_max = self._hf.get_z_mid_z_max_from_mesh()
        self._wl = Waterline(z_mid,self.xmf)
        res = optimize.root(self.fun_opt_root_drought_calculation, z_mid)
        print(res)
        x=res['x']
        pp_z=x[0]
        self._wl.set_plane_point_z(pp_z)
        print('pp_z = ',pp_z)
        fvs = self._mesh.fv_indices().tolist()
        points = self._mesh.points().tolist()
        print('T = ',self.wl.get_max_distance_to_mesh(fvs, points)[0])


    def determine_trim_for_current_load_condition(self):
        #self.wl.reset_to_heel_angle_deg(20)
        x0 = np.array([self.wl.point[2],self.wl.normal[0]])
        res = optimize.root(self.fun_opt_root_trim_calculation, x0)
        x=res['x']
        pp_z = x[0]
        pl_nv_x = x[1]
        self.wl.set_plane_point_z(pp_z)
        self.wl.set_plane_normal_x(pl_nv_x)
        fvs = self._mesh.fv_indices().tolist()
        points = self._mesh.points().tolist()
        displacement, displacementCG, new_fvs, new_pts = self.calculate_displacement_and_displacementCG()
        displacement =  displacement * self._sea_density / 1000.0
        print('trim, deg = ', self.wl.get_trim_angle())
        print('displacement, t = ', displacement)
        if False: # Print additional data
            print('T = ', self.wl.get_max_distance_to_mesh(fvs, points))
            print(self.wl.point)
            print(self.wl.normal)
        if True: # Create geometries for visualization
            testgeo = GeometryExtension('Trim mesh')
            testgeo.mesh = om.TriMesh(new_pts, new_fvs)
            testgeo.emit_geometry_built()
            bb = self._hf.bbox
            length = bb.maxCoord[0] - bb.minCoord[0]
            width = bb.maxCoord[1] - bb.minCoord[1]
            self.wl.create_and_emmit_geometry('WL heel={0};-trim={1}'.format(self.wl.get_heel_angle(),self.wl.get_trim_angle()),length,width)

    def fun_opt_root_drought_calculation(self, pp_z: float):
        diff = self._ship_weight - self.calculate_displacement_horizontal_waterline(pp_z)*self._sea_density/1000.0
        return diff

    def fun_opt_root_trim_calculation(self, x):
        pp_z = x[0]
        pl_nv_x = x[1]
        self._wl.set_plane_point_z(pp_z)
        self._wl.set_plane_normal_x(pl_nv_x)
        displacement, displacementCG, new_fvs, new_pts = self.calculate_displacement_and_displacementCG()
        diff1 = self._ship_weight - displacement*self._sea_density/1000.0
        # Racunanje projekcija tezista istisnine i tezista broda na ravninu vodne linije
        displacementCG_proj = self.project_point_on_plane(displacementCG)
        shipCG_proj = self.project_point_on_plane(self._ship_CG)
        # Racunanje razlike izmedu projekcija tezista istisnine i tezista broda
        diff2 = np.linalg.norm(displacementCG_proj - shipCG_proj)
        return diff1,diff2

    def project_point_on_plane(self, point):
        plane_point = self._wl.point
        plane_normal = self._wl.normal
        d = -plane_normal.dot(plane_point)
        t = -(plane_normal.dot(point) + d) / (plane_normal.dot(plane_normal))
        projection = point + t * plane_normal
        return projection

    def calculate_displacement_horizontal_waterline(self,pp_z :float):
        fvs = self._mesh.fv_indices().tolist()
        points = self._mesh.points().tolist()
        plane_point = self._wl.point
        plane_normal = self._wl.normal
        plane_point[2] = pp_z
        new_fvs, new_pts = self.get_mesh_below_inclined_waterline(fvs, points, plane_point, plane_normal)
        displacement, centroid = self.calculate_displacement_and_centroid(new_fvs, new_pts)
        return displacement

    def set_ship_G_and_CG(self,G:float,xCG,yCG,zCG):
        self._ship_weight: float = G
        self._ship_CG[0]= xCG
        self._ship_CG[1] = yCG
        self._ship_CG[2] = zCG

    def generate_static_stability_curve(self, max_heel_angle:float, step_heel_angle:float) -> (np.ndarray,np.ndarray):
        heeling_angles = np.arange(0.0, max_heel_angle+step_heel_angle, step_heel_angle).tolist()
        fvs = self._hf.mesh.fv_indices().tolist()
        pts = self._hf.mesh.points().tolist()
        bb= self._hf.bbox
        length = bb.maxCoord[0]-bb.minCoord[0]
        width = bb.maxCoord[1] - bb.minCoord[1]
        z_mid, z_max = self._hf.get_z_mid_z_max_from_mesh()
        stability_curve_data=[]
        i=0
        for heel in heeling_angles:
            heeling_angle = math.radians(heel)
            self._wl.reset_to_heel_angle_deg(heel)
            #if self.wl.point[2] < z_mid/5:
            #    self.wl.set_plane_point_z(z_mid/5)
            self.wl.set_plane_point_z(z_mid)
            try:
                self.determine_trim_for_current_load_condition()
                displacement, CB,new_fvs, new_pts = self.calculate_displacement_and_displacementCG()
                GZ = 0
                if i > 0:
                    CB_proj = self.project_point_on_plane(CB)
                    ship_cg_proj = self.project_point_on_plane(self._ship_CG)
                    vd = np.cross(self.wl.normal,(CB-self._ship_CG))
                    GZ=np.linalg.norm(vd)
                    if vd[1] > 0:
                        GZ =-GZ
                T = self._wl.get_max_distance_to_mesh(new_fvs, new_pts)[0]
                #self.wl.create_and_emmit_geometry('Point {0}: heel {1:3.0f},°; GZ {2:6.3f}'.format(i + 1, heel, GZ),length,width)
                # if T > z_max*1.1:
                #     print(
                #         'Point {0}: heel {1:3.0f},°; GZ {2:6.3f} ; out heel {3:6.2f},°; trim {4:6.2f},°; T {5:6.2f}, m'.format(
                #             i + 1, heel, GZ, self._wl.get_heel_angle(), self._wl.get_trim_angle(), T))
                #     print('Further calculation of stability curve stopped, Drought to high!')
                #     continue
                displacement =displacement*self._sea_density/1000.0
                stability_curve_data.append([heel, GZ, self._wl.get_trim_angle(),displacement,T, self._wl.get_heel_angle()])
                print(
                    'Point {0}: heel {1:3.0f},°; GZ {2:6.3f} ; out heel {3:6.2f},°; trim {4:6.2f},°; T {5:6.2f}, m'.format(
                        i + 1, heel, GZ, self._wl.get_heel_angle(), self._wl.get_trim_angle(),T))
            except BaseException as error:
                print('Point {0}: heel {1:3.0f},°; ERROR: {2}'.format(i + 1, heel,error))
                print('Further calculation of stability curve stopped!')
                continue
            except:
                print('Unknown exception occurred during right arm calculation')
            i+=1
        return stability_curve_data


    def point_distance(self, a, b):
        return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)

    def get_mesh_below_inclined_waterline_numpy\
                    (self, fvs:List[np.ndarray],points:List[np.ndarray],plane_point:np.ndarray,plane_normal:np.ndarray):
        #TODO
        return None, None

    def get_mesh_below_inclined_waterline\
                    (self, fvs:List[np.ndarray],points:List[np.ndarray],plane_point:np.ndarray,plane_normal:np.ndarray):
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
        new_cp= Waterline.get_point_on_plane_close_to_input_point(plane_point,plane_normal,cp)
        cp[:] = new_cp[:]
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

    @property
    def xmf(self):
        return  self._xmf

def test_wl_1():
    wl1 = Waterline(5)
    wl1.reset_to_heel_angle_deg(181)
    print(wl1._plane_normal)
    angle = wl1.get_angle_between_plane_normal_and_ref_plane_normal()
    print(angle)

def test_wl_2():
    wl = Waterline(5)
    wl.reset_to_heel_angle_deg(20)
    wl.normal[0]=0.05
    print(wl.normal)
    w = np.array([-1.0,0.0,0.0])
    u= np.cross(w,wl.normal)
    v = np.cross(wl.normal,u)
    print(v)

def create_box():
        dx=0
        dy = 0
        dz = 0
        p=[]
        p.append(np.array([-1+dx, -1+dy, -1+dz])) # 0
        p.append(np.array([-1+dx, -1+dy,  1+dz])) # 1
        p.append(np.array([-1+dx,  1+dy, -1+dz])) # 2
        p.append(np.array([-1+dx,  1+dy,  1+dz])) # 3
        p.append(np.array([ 1+dx, -1+dy, -1+dz])) # 4
        p.append(np.array([ 1+dx, -1+dy,  1+dz])) # 5
        p.append(np.array([ 1+dx,  1+dy, -1+dz])) # 6
        p.append(np.array([ 1+dx,  1+dy,  1+dz])) # 7
        fvi = []
        fvi.append(np.array([0, 6, 4]))
        fvi.append(np.array([0, 2, 6]))

        fvi.append(np.array([0, 4, 5]))
        fvi.append(np.array([0, 5, 1]))

        fvi.append(np.array([0, 3, 2]))
        fvi.append(np.array([0, 1, 3]))

        fvi.append(np.array([6, 2, 3]))
        fvi.append(np.array([6, 3, 7]))

        fvi.append(np.array([4, 7, 5]))
        fvi.append(np.array([4, 6, 7]))

        fvi.append(np.array([1, 5, 7]))
        fvi.append(np.array([1, 7, 3]))
        return p,fvi
def test_numpy_mbw():
    points,fvi = create_box()
    plane_normal=np.array([0.0, 0.0, -1])
    plane_point = np.array([0.0, 0.0, 0.5])

    sscalc = ShipStability(None,0)
    sscalc.get_mesh_below_inclined_waterline_numpy(fvi, points, plane_point, plane_normal)

if __name__ == "__main__":
    # Test
    test_numpy_mbw()