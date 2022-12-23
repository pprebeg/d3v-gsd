from core.geometry import Geometry
from core import geometry_manager as manager
from typing import Dict
key_show_face = 'show_face'
key_show_edge = 'show_edge'
class GeometryExtension(Geometry):
    def __init__(self,name='', guid=None):
        super().__init__(name,guid)
        self._show_mesh_edges=True
        self._show_mesh_faces = True
        self._geometry_attributes:Dict[str,object] = {key_show_face:True,key_show_edge:True}


    @property
    def attributes(self):
        return self._geometry_attributes

    @property
    def show_mesh_faces(self):
        if key_show_face in self._geometry_attributes:
            return self._geometry_attributes[key_show_face]
        return True

    @property
    def show_mesh_edges(self):
        if key_show_edge in self._geometry_attributes:
            return self._geometry_attributes[key_show_edge]
        return True

    @show_mesh_faces.setter
    def show_mesh_faces(self, value):
        self._geometry_attributes[key_show_face] = bool(value)

    @show_mesh_edges.setter
    def show_mesh_edges(self, value):
        self._geometry_attributes[key_show_edge] = bool(value)

    def have_mesh(self):  # override for better efficiency
        return self.numfaces > 0

    @property
    def num_faces(self):
        return self.mesh.n_faces()

    def emit_geometries_rebuild(self):
        manager.remove_geometry([self])
        manager.add_geometry([self])
        manager.show_geometry([self])
    def emit_geometry_built(self):
        manager.add_geometry([self])
        manager.show_geometry([self])
