from extendedgeometry import ExtendedGeometry
import openmesh as om
import os
import numpy as np
import copy
import csv
from typing import List, Dict,Tuple
from hullformdir.hullform import HullForm

class HstarBase(ExtendedGeometry):
	def __init__(self, id):
		super().__init__()
		self._position = 0
		self._size = 0
		self._id = id
		self._show = True
		self._do_portside = True
		self._do_starbside = True

	@property
	def show(self):
	    return self._show

	def set_show(self,do_show):
		before = self.show
		self._show = do_show
		return (before == self.show)

	@property
	def id(self):
		return self._id

	@property
	def position(self):
		return self._position

	@property
	def size(self):
		return self._size
	@staticmethod
	def join_tria_meshes(mesh1:om.TriMesh,mesh2:om.TriMesh):
		fvs1 = mesh1.fv_indices()
		points1 = mesh1.points()
		fvs2 = mesh2.fv_indices()
		points2 = mesh2.points()
		npts1 = points1.shape[0]
		fvs2 = fvs2+npts1
		fvs = np.concatenate((fvs1,fvs2), axis=0)
		points = np.concatenate((points1,points2), axis=0)
		return om.TriMesh(points,fvs)

	@staticmethod
	def	get_facet_condition_satisfaction(fv_points:[],min_points:int,is_inverse:bool,i_test_dim:[],tol:float,lbs:[],ubs:[]=None):

		if min_points==0: #use cg
			min_points =1
			p = np.zeros(3)
			for point in fv_points:
				p += point
			p=p/len(fv_points)
			points = [p]
		elif min_points == -1:  # use all
			min_points = len(fv_points)
			points = fv_points
		else:
			points = fv_points
		num_points = 0
		for point in points:
			is_in = True
			for i_xyz in range(len(i_test_dim)):
				i_dim = i_test_dim[i_xyz]
				xyz = point[i_dim]
				# area inside the belt between lower and upper bound
				if not is_inverse:
					if xyz < (lbs[i_xyz]+tol):
						is_in = False
						break
					if ubs is not None:
						if xyz > (ubs[i_xyz]-tol) :
							is_in = False
							break
				# area without the belt between lower and upper bound
				else:
					if xyz > (lbs[i_xyz] - tol):
						is_in = False
						break
					if ubs is not None:
						if xyz < (ubs[i_xyz] + tol):
							is_in = False
							break
			if is_in:
				num_points+=1
		return (num_points >= min_points)

	@staticmethod
	def cut_mesh_with_plane(mesh:om.PolyMesh,
							min_points:int,is_inverse:bool,i_test_dim:[],tol:float,lbs:[],ubs:[]=None):
		fvs = mesh.fv_indices()
		points = mesh.points()
		new_fvs = []
		old_to_new_point={}
		i_v = 0
		fv_points = []
		for fv in fvs:
			is_in = True
			fv_points.clear()
			for i in fv:
				fv_points.append(points[i])
			is_in = HstarBase.get_facet_condition_satisfaction(fv_points, min_points, is_inverse, i_test_dim, tol, lbs, ubs)
			if is_in:
				new_fv = []
				for i in fv:
					if i not in old_to_new_point:
						old_to_new_point[i]=i_v
						i_v+=1
					new_fv.append(old_to_new_point[i])

				new_fvs.append(np.array(new_fv))
		new_points = [0]*len(old_to_new_point)
		for i_old,i_new in old_to_new_point.items():
			new_points[i_new]=points[i_old]

		new_points = np.array(new_points)
		new_fvs = np.array(new_fvs)
		new_mesh = om.TriMesh(new_points, new_fvs)
		return new_mesh


	def get_info(self) -> str:
		msg = '\nid = ' + str(self.id)
		msg += '\nposition = ' + str(self.position)
		msg += '\ndimensions = ' + str(self.size)
		return msg

	def regenerateMesh(self):
		self._genMesh()

	def _update_mesh_on_show(self):
		pass

	def regenerate_mesh_port_starboard(self,do_portside:bool = True, do_starboardside:bool = True):
		self._do_portside = do_portside
		self._do_starbside = do_starboardside
		if (not do_portside) and (not do_starboardside):
			self.mesh = om.TriMesh()
		else:
			self.regenerateMesh()
			self.post_gen_mesh()

	def post_gen_mesh(self):
		#handle portside starboard side
		lbs = [0.0]
		i_test_dims = [1]
		tol = 0.0
		if not self._do_starbside:
			self.mesh = HstarBase.cut_mesh_with_plane(self.mesh, -1, False, i_test_dims, tol, lbs)
		if not self._do_portside:
			self.mesh = HstarBase.cut_mesh_with_plane(self.mesh, -1, True, i_test_dims, tol, lbs)

	def set_position(self, new_position):
		move_vector = new_position - self.position
		self.move(move_vector)

	def _genMesh(self):
		pass

class HstarHullForm(HstarBase):
	def __init__(self, fileName):
		super().__init__(0)
		self._hullform:HullForm = HullForm(fileName)
		self._position = np.array([0., 0., 0.])
		self._centroid = np.array([self.LOA / 2, 0, 0])  # sredina samo za x za sada
		self.mesh = self._hullform.mesh

	@property
	def deck_indexes(self):
	    return self._hullform.dict_decks


	def get_deck_z_index(self,deck_index):
		return self._hullform.pdecks[deck_index]

	def get_deck_z_code(self,deck_code):
		deck_index = self._hullform.dict_decks.get(deck_code)
		return self.get_deck_z_index(deck_index)

	@property
	def centroid(self):
	    return self._centroid

	@property
	def LOA(self):
	    return self._hullform.shipdata["loa_val"]

	@property
	def BOA(self):
		return self._hullform.shipdata["boa_val"]

	@property
	def D(self):
		return self._hullform.shipdata["draft_val"]

	@property
	def name(self):
		return self._hullform.shipdata["shipname"]

	def _genMesh(self):
		pass


	def get_info(self) -> str:
		msg="Ship: "+self.name
		msg += "\nLOA = "+str(self.LOA)
		msg += "\nBOA = " + str(self.BOA)
		msg += "\nD = " + str(self.D)
		return msg

class HstarProblem():
	def __init__(self, fileName):
		self._filename = fileName
		self._num_frames = 20
		if (fileName != ""):
			self._hull:HstarHullForm = self.read_problem()

	@property
	def hull(self):
	    return self._hull

	@property
	def num_frames(self):
	    return self._num_frames

	@num_frames.setter
	def num_frames(self, new_value):
		self._num_frames = new_value

	def read_problem(self):
		with open(self._filename, "r") as csv_file:
			csv_reader = csv.DictReader(csv_file)
			for row in csv_reader:  # each row contains 1 block data
				hull_form_input = str(row["hull_form_file"])
				n_fr = int(row["num_frames"])

			abspath1 = '\\'.join(self._filename.split('\\')[0:-1])
			abspath2 = '/'.join(self._filename.split('/')[0:-1])
			if len(abspath2) > len(abspath1):
				sf_char = '/'
				abspath = abspath2 + '/'
				hull_form_input = hull_form_input.replace('\\', '/')
			else:
				sf_char = '\\'
				abspath = abspath1 + '\\'
				hull_form_input = hull_form_input.replace('/', '\\')

			hull_form_input = abspath + hull_form_input

			return HstarHullForm(hull_form_input)