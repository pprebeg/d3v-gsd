from geometry_extend import GeometryExtension
import openmesh as om
import numpy as np
import os

class HullForm(GeometryExtension):
    def __init__(self, fileName,name='',translation=np.zeros(3)):
        super().__init__(name)
        self._filename = fileName
        self._translation=translation

    @property
    def filename(self):
        return self._filename

    def regenerateHullHorm(self):
        pass

    def get_x_main_frame_from_mesh(self):
        if self.mesh is not None:
            bb=self.bbox
            xmin = bb.minCoord[0]
            xmax = bb.maxCoord[0]
            return (xmax + xmin)/2.0
        return 0
    def get_z_mid_z_max_from_mesh(self):
        if self.mesh is not None:
            bb=self.bbox
            zmin = bb.minCoord[2]
            zmax = bb.maxCoord[2]
            return (zmax + zmin)/2.0, zmax
        return 0,0

    def miror_mesh(self):
        if self.mesh is not None:
            bb=self.bbox
            xmin = bb.minCoord[0]
            xmax = bb.maxCoord[0]
            fvi = self.mesh.fv_indices()
            points = self.mesh.points()
            points[:,0] = xmax-points[:,0]+xmin
            self.mesh = om.TriMesh(points, fvi)

    def miror_mesh_around_simetry_plan(self):
        if self.mesh is not None:
            bb=self.bbox
            xmin = bb.minCoord[0]
            xmax = bb.maxCoord[0]
            fvi = self.mesh.fv_indices()
            points = self.mesh.points()
            points2= points.copy()
            points2[:,1]= - points2[:,1]
            lpoints = list(points)
            fvi2=fvi.copy()
            fvi2=fvi2+len(lpoints)
            lfvi = list(fvi)
            lfvi.extend(list(fvi2))
            lpoints.extend(list(points2))

            self.mesh = om.TriMesh(lpoints, lfvi)

    def rise_mesh_ends(self,rise_bow,rise_stern):
        if self.mesh is not None:
            bb=self.bbox
            xmin = bb.minCoord[0]
            xmax = bb.maxCoord[0]
            fvi = self.mesh.fv_indices()
            points = self.mesh.points()
            xmid = (xmin + xmax)/2.0
            points[:,2] = np.where(points[:,0 ]> xmid,
                                   points[:,2]+ np.power(np.abs(xmid - points[:,0])/(xmax-xmin)*1.5,2.0)*rise_bow,
                                   points[:,2])
            points[:, 2] = np.where(points[:, 0] < xmid,
                                    points[:, 2] + np.power(np.abs(xmid - points[:, 0]) / (xmax - xmin) * 1.5,
                                                            2.0) * rise_stern,
                                    points[:, 2])
            self.mesh = om.TriMesh(points, fvi)

    def translate(self,translate_vector):
        translate_vector=np.array(translate_vector)
        self._translation += translate_vector
        self.regenerateHullHorm()

    def exportGeometry(self, fileName):
        pass

class HullFormFromMesh(HullForm):
    def __init__(self, fileName,name='',translation=np.zeros(3)):
        super().__init__(fileName,name,translation)
        self._translation =np.array([0.0,0.0,0.0]) # kyrinia example
        self._original_mesh =self.read_file()
        self.regenerateHullHorm()

    @staticmethod
    def export_hull_form(fileName:str,hull_form:HullForm):
        filename_no_ext, file_extension = os.path.splitext(fileName)
        if file_extension == ".stl" or file_extension == ".obj":
            om.write_mesh(fileName,hull_form.mesh)

    def read_file(self):
        return om.read_trimesh(self.filename)

    def regenerateHullHorm(self):
        if self._original_mesh is not None:
            fvi = self._original_mesh.fv_indices().copy()
            points = self._original_mesh.points().copy()
            points+=self._translation
            self.mesh =om.TriMesh(points, fvi)
            #self.miror_mesh() # kyrinia

    def exportGeometry(self, fileName):
        HullFormFromMesh.export_hull_form(fileName,self)


