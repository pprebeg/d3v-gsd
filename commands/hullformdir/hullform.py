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

    def get_length_from_mesh(self):
        if self.mesh is not None:
            points = self.mesh.points()
            xmin = points.min(axis=0)[0]
            xmax = points.max(axis=0)[0]
            return xmax - xmin
        return 0

    def translate(self,translate_vector):
        translate_vector=np.array(translate_vector)
        self._translation += translate_vector
        self.regenerateHullHorm()

    def exportGeometry(self, fileName):
        pass

class HullFormFromMesh(HullForm):
    def __init__(self, fileName,name='',translation=np.zeros(3)):
        super().__init__(fileName,name,translation)
        self._translation =np.array([1.0,0.0,0.0]) # kyrinia example
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
            fvi = self._original_mesh.fv_indices()
            points = self._original_mesh.points()
            points+=self._translation
            self.mesh =om.TriMesh(points, fvi)

    def exportGeometry(self, fileName):
        HullFormFromMesh.export_hull_form(fileName,self)


