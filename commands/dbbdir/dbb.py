import openmesh as om
import os
import numpy as np
import copy
import csv
from scipy.spatial.transform import Rotation as R

from hullformdir.hullform import HullForm
#d3v imports
from extendedgeometry import ExtendedGeometry
from typing import List, Dict,Tuple
import itertools
from collections import deque
use_gmsh_cut = True
if use_gmsh_cut:
	import dbbdir.dbbcutgmsh as mf
else:
	import dbbdir.dbbcut as mf


class DBBBaseAll(ExtendedGeometry):
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

	@position.setter
	def position(self, value):
	    self._position = value

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
			is_in = DBBBaseAll.get_facet_condition_satisfaction(fv_points,min_points,is_inverse,i_test_dim,tol,lbs,ubs)
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
			self.mesh = DBBBaseAll.cut_mesh_with_plane(self.mesh, -1, False, i_test_dims, tol, lbs)
		if not self._do_portside:
			self.mesh = DBBBaseAll.cut_mesh_with_plane(self.mesh, -1, True, i_test_dims, tol, lbs)

	def move(self, move_vector):
		self.position += move_vector
		self.mesh = mf.move_mesh(self.mesh, move_vector)

	def set_position(self, new_position):
		move_vector = new_position - self.position
		self.move(move_vector)

	def _genMesh(self):
		pass


class DBBHullFormAll(DBBBaseAll):
	def __init__(self, fileName):
		super().__init__(0)
		self._hullform = HullForm(fileName)
		self.mesh = self._hullform.mesh
		self._position = np.array([0., 0., 0.])
		self._centroid = np.array([self.LOA / 2, 0, 0])  # sredina samo za x za sada

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
		self.mesh = self._hullform.mesh

	def move(self, move_vector):
		self.position += move_vector
		self.mesh = mf.move_mesh(self.mesh, move_vector)

	def get_info(self) -> str:
		msg="Ship: "+self.name
		msg += "\nLOA = "+str(self.LOA)
		msg += "\nBOA = " + str(self.BOA)
		msg += "\nD = " + str(self.D)
		return msg

class DBBBulkhead(DBBBaseAll):
	def __init__(self, bulk_id,bulk_type:str, hullform:DBBHullFormAll, size: np.ndarray=None, position: np.ndarray=None):
		super().__init__(bulk_id)
		self._hullform:DBBHullFormAll = hullform
		self._size= size
		self._position = position
		self._bulk_type = bulk_type
		self.regenerateMesh()

	show_bulkheads = True

	@property
	def hullform(self):
	    return self._hullform._hullform

	@property
	def x_frame(self):
		return self.position[0]

	@property
	def z_start(self):
		return self.position[2]

	@property
	def z_end(self):
		return self.position[2]+self._size[2]

	@DBBBaseAll.show.getter
	def show(self):
		return self._show and DBBBulkhead.show_bulkheads

	def gen_mesh_points_fvs(self,frame_points:List[np.ndarray]):
		points = frame_points.copy()
		fvs=[]
		n_p= len(frame_points)
		isym_1=0
		if frame_points[0][1] != 0.0:
			sym_p =np.array(frame_points[0])
			sym_p[1] = 0.0
			isym_1 = len(points)
			points.append(sym_p)

		for i in range(1,n_p):
			sym_p = np.array(frame_points[i])
			sym_p[1] = 0.0
			isym_2 = len(points)
			points.append(sym_p)
			if isym_1 >=n_p:
				fvs.append(np.array([i-1, isym_1, isym_2]))
			fvs.append(np.array([i-1, isym_2, i]))
			isym_1=isym_2
		return points,fvs

	def trim_sorted_points_by_z_start_end(self,points):
		trim_points = []
		n_p=len(points)
		z_tol = 0.0001
		i=0
		while(i < n_p and points[i][2]<self.z_start):
			i+=1
		if i > 0 and (points[i][2]-z_tol) > self.z_start:
			p = np.array([self.x_frame,0.0,self.z_start])
			p[1]=points[i-1][1]+(points[i][1]-points[i-1][1])/(points[i][2]-points[i-1][2])*(self.z_start-points[i-1][2])
			trim_points.append(p)
		while (i < n_p and points[i][2] <= self.z_end):
			trim_points.append(points[i])
			i += 1
		if i > 0 and i < n_p and (points[i-1][2]+z_tol) < self.z_end:
			p =  np.array([self.x_frame,0.0,self.z_end])
			p[1]=points[i-1][1]+(points[i][1]-points[i-1][1])/(points[i][2]-points[i-1][2])*(self.z_end-points[i-1][2])
			trim_points.append(p)

		return trim_points

	def getSortedPointsOnFrame(self, x, fvs, points):
		axis_index=0
		points_on_plane = []
		lpr = []
		lpl = []
		for fv in fvs:
			lpr.clear()
			lpl.clear()
			for iv in fv:
				p = points[iv]
				if p[axis_index] < x:
					lpl.append(iv)
				elif p[axis_index] > x:
					lpr.append(iv)
				else:
					points_on_plane.append(p)

			if len(lpl) > 0 and len(lpr) > 0:
				if len(lpl) < len(lpr):
					points_on_plane.append(self.getIntersectionPoint(points[lpl[0]], points[lpr[0]], x, axis_index))
					points_on_plane.append(self.getIntersectionPoint(points[lpl[0]], points[lpr[1]], x, axis_index))
				elif len(lpl) > len(lpr):
					points_on_plane.append(self.getIntersectionPoint(points[lpl[0]], points[lpr[0]], x, axis_index))
					points_on_plane.append(self.getIntersectionPoint(points[lpl[1]], points[lpr[0]], x, axis_index))
				else:
					points_on_plane.append(self.getIntersectionPoint(points[lpl[0]], points[lpr[0]], x, axis_index))
				pass


		points_on_plane.sort()
		temp_points = list(points_on_plane for points_on_plane, _ in itertools.groupby(points_on_plane))
		points_on_plane = temp_points
		points_on_plane = sorted(points_on_plane, key=lambda p: p[2])
		points_pos = []
		points_neg = []
		for p in points_on_plane:
			if p[1] > 0:
				points_pos.append(p)
			elif p[1] < 0:
				points_neg.append(p)
			else:
				points_pos.append(p)
				points_neg.append(p)
		return points_pos,points_neg

	def getIntersectionPoint(self, p1, p2, h, axis_index):
		ip1 = 0
		if axis_index == 2:
			ip1 = [(h - p2[2]) / (p1[2] - p2[2]) * (p1[0] - p2[0]) + p2[0],
				   (h - p2[2]) / (p1[2] - p2[2]) * (p1[1] - p2[1]) + p2[1], h]
		if axis_index == 0:
			ip1 = [h, (h - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1],
				   (h - p1[0]) / (p2[0] - p1[0]) * (p2[2] - p1[2]) + p1[2]]

		return ip1
	def _genMesh(self):  # za keel koji je na 0 nema wl

		meshp = None
		meshn = None
		fvs = self.hullform.mesh.fv_indices().tolist()
		points = self.hullform.mesh.points().tolist()
		points_pos,points_neg = self.getSortedPointsOnFrame(self.x_frame, fvs, points)
		points_pos = self.trim_sorted_points_by_z_start_end(points_pos)
		points_neg = self.trim_sorted_points_by_z_start_end(points_neg)
		pneg,fvsneg = self.gen_mesh_points_fvs(points_neg)
		ppos, fvspos = self.gen_mesh_points_fvs(points_pos)
		meshp = om.TriMesh(ppos, fvspos)
		meshn = om.TriMesh(pneg, fvsneg)


		if meshn is not None and meshp is not None:
			self.mesh = DBBBaseAll.join_tria_meshes(meshp,meshn)
		else:
			self.mesh = om.TriMesh()
	def get_info(self) -> str:
		msg= super().get_info()
		msg="Bulkhead"+msg
		return msg

class DBBDeckBase(DBBBaseAll):
	def __init__(self,deck_id, hullform:DBBHullFormAll, size: np.ndarray=None, position: np.ndarray=None):
		super().__init__(deck_id)
		self._hullform:DBBHullFormAll = hullform
		self._size= size
		self._position = position

	@property
	def hullform(self):
	    return self._hullform._hullform
	@property
	def z(self):
		return self.position[2]

class DBBDeckUnitBase(DBBDeckBase):
	def __init__(self, deck_id, hullform, size: np.ndarray, position: np.ndarray, parent_deck_segment: DBBBaseAll):
		super().__init__(deck_id, hullform, size, position)
		self._deck_segment: DBBBaseAll = parent_deck_segment

	@property
	def deck_segment(self):
	    return self._deck_segment

class DBBDeckPlate(DBBDeckUnitBase):
	def __init__(self,deck_id, hullform,  size: np.ndarray, position: np.ndarray,parent_deck_segment:DBBBaseAll):
		super().__init__(deck_id, hullform, size, position,parent_deck_segment)
		self.regenerateMesh()

	show_deck_plate = True

	@DBBBaseAll.show.getter
	def show(self):
	   return self._show and DBBDeckPlate.show_deck_plate

	def _genMesh(self):  # za keel koji je na 0 nema wl
		deck_points = None
		meshp = None
		for wline in self.hullform.wlinesPos:
			if np.isclose(self.z, wline[0][2]):
				deck_points = np.asarray(wline)
				break
		if deck_points is not None:
			meshp = mf.make_deck(deck_points, subdivide=False)
		deck_points = None
		meshn = None
		for wline in self.hullform.wlinesNeg:
			if np.isclose(self.z, wline[0][2]):
				deck_points = np.asarray(wline)
				break
		if deck_points is not None:
			meshn = mf.make_deck(deck_points, subdivide=False)

		if meshn is not None and meshp is not None:
			self.mesh = DBBBaseAll.join_tria_meshes(meshp,meshn)
		else:
			self.mesh = om.TriMesh()
	def get_info(self) -> str:
		msg= super().get_info()
		msg="Deck Plate"+msg
		return msg
class DBBDeckPlateRSB(DBBDeckPlate):
	def __init__(self,deck_id, hullform,  size: np.ndarray, position: np.ndarray,parent_deck_segment:DBBBaseAll,
				 rsb_deck_points: List[List[np.ndarray]]):
		self._rsb_deck_points:List[List[np.ndarray]]=rsb_deck_points
		super().__init__(deck_id, hullform, size, position,parent_deck_segment)

	def _gen_mesh_points_fvs(self, deck_points:List[np.ndarray]):
		#points are prepared to start and end with y=0
		points = deck_points.copy()
		fvs=[]
		n_p= len(deck_points)
		isym_1=0
		for i in range(1,n_p-1):
			sym_p = np.array(deck_points[i])
			sym_p[1] = 0.0
			isym_2 = len(points)
			points.append(sym_p)
			if isym_1 >=n_p:
				fvs.append(np.array([i-1, isym_1, isym_2]))
			fvs.append(np.array([i-1, isym_2, i]))
			isym_1=isym_2
		fvs.append(np.array([i - 1, isym_1, i]))
		return points,fvs
	def _genMesh(self):  # za keel koji je na 0 nema wl
		self.mesh = None
		for rsb_point in self._rsb_deck_points:
			result = self._gen_pos_neg_points(rsb_point)
			for points in result:
				points,fvs = self._gen_mesh_points_fvs(points)
				mesh = om.TriMesh(points, fvs)
				if self.mesh is None:
					self.mesh = mesh
				else:
					self.mesh = DBBBaseAll.join_tria_meshes(self.mesh, mesh)


	def _gen_pos_neg_points(self, rsb_point):
		points_pos = []
		points_neg = []
		is_last_pos_neg = False
		is_last_neg_pos = False
		ilast = len(rsb_point)-1
		if rsb_point[0][0]==rsb_point[ilast][0] and rsb_point[0][1]==rsb_point[ilast][1]:
			ilast=ilast-1
		for i in range(len(rsb_point)):
			p = rsb_point[i]
			if np.allclose(rsb_point[i],rsb_point[ilast],0.001):
				continue
			if p[1] > 0:
				if is_last_pos_neg:
					pnew = np.array([0.0, 0.0, p[2]])
					pnew[0] = rsb_point[ilast][0] + (rsb_point[i][0] - rsb_point[ilast][0]) / (
							rsb_point[i][1] - rsb_point[ilast][1]) * (0.0 - rsb_point[ilast][1])
					points_pos.append(pnew)
					points_neg.append(pnew)
				points_pos.append(p)
				is_last_pos_neg = False
				is_last_neg_pos = True
			elif p[1] < 0:
				if is_last_neg_pos:
					pnew = np.array([0.0, 0.0, p[2]])
					pnew[0] = rsb_point[ilast][0] + (rsb_point[i][0] - rsb_point[ilast][0]) / (
						rsb_point[i][1] - rsb_point[ilast][1]) * (0.0 - rsb_point[ilast][1])
					points_neg.append(pnew)
					points_pos.append(pnew)
				points_neg.append(p)
				is_last_pos_neg = True
				is_last_neg_pos = False
			else:
				points_pos.append(p)
				points_neg.append(p)
				is_last_pos_neg = False
				is_last_neg_pos = False
			ilast = i
		# handle last point
		p1 = rsb_point[0]
		pnew = None
		if (p1[1] * p[1]) < 0:
			pnew = np.array([0.0, 0.0, p[2]])
			pnew[0] = p[0] + (p1[0] - p[0]) / (p1 - p[1]) * (0.0 - p[1])
			if p[1] < 0:
				points_pos.insert(0, pnew)
				points_neg.append(pnew)
			elif p[1] > 0:
				points_neg.insert(0, pnew)
				points_pos.append(pnew)
			else:
				if p1[1] > 0:
					points_pos.insert(0, pnew)
				elif p1[1] < 0:
					points_neg.insert(0, pnew)
				elif p[1] < 0:
					points_neg.append(pnew)
				elif p[1] > 0:
					points_pos.append(pnew)
		if p1[1] < 0: # started with negative y values
			temp = points_neg.copy()
			# find second zero
			n0=0
			i=0
			for p in temp:
				if p[1] == 0.0:
					n0+=1
					if n0 == 2:
						break
				i+=1
			ineg=0
			for j in range(i,len(temp)):
				points_neg[ineg]=temp[j]
				ineg+=1
			for j in range(0,i):
				points_neg[ineg]=temp[j]
				ineg+=1
		# elif not np.allclose(p1,p,0.001):
		# 	if p[1] > 0:
		# 		points_pos.append(p1)
		# 	elif p[1] > 0:
		# 		points_pos.append(p1)
		# 	else:
		# 		if p1[1] > 0:
		# 			points_pos.insert(0, p)
		# 			points_neg.append(p1)
		# 		elif p1[1] < 0:
		# 			points_neg.insert(0, p)
		# 			points_pos.append(p1)
		return points_pos,points_neg

	def get_info(self) -> str:
		msg= super().get_info()
		msg="Deck Plate RSB"+msg
		return msg
class DBBDeckHull(DBBDeckUnitBase):
	def __init__(self, deck_id, hullform, size: np.ndarray, position: np.ndarray, parent_deck_segment: DBBBaseAll):
		super().__init__(deck_id, hullform, size, position,parent_deck_segment)
		self.regenerateMesh()

	show_deck_hull = True

	@DBBBaseAll.show.getter
	def show(self):
	   return self._show and DBBDeckHull.show_deck_hull

	def _genMesh(self):  # za keel koji je na 0 nema wl
		zlb = self.position[2]
		zub = zlb+self.size[2]
		lbs=[zlb]
		ubs =[zub]
		i_test_dims = [2]
		self.mesh = DBBBaseAll.cut_mesh_with_plane(self._hullform.mesh,-1,False,i_test_dims,0.0,lbs,ubs)
	def get_info(self) -> str:
		msg= super().get_info()
		msg="Deck Hull"+msg
		return msg
class DBBDeckHullRSB(DBBDeckHull):
	def __init__(self, deck_id, hullform, size: np.ndarray, position: np.ndarray, parent_deck_segment: DBBBaseAll,
				 rsb_hull_points:List[List[List[np.ndarray]]]):
		self._rsb_hull_points:List[List[List[np.ndarray]]]=rsb_hull_points

		super().__init__(deck_id, hullform, size, position,parent_deck_segment)



	def _genMesh(self):  # za keel koji je na 0 nema wl
		self.mesh = None
		for rsb_points in self._rsb_hull_points:
			points,fvs = self._gen_mesh_points_fvs(rsb_points)
			mesh = om.TriMesh(points, fvs)
			if self.mesh is None:
				self.mesh = mesh
			else:
				self.mesh = DBBBaseAll.join_tria_meshes(self.mesh,mesh)

	def _gen_mesh_points_fvs(self,ulpoints:List[np.ndarray]):
		points = ulpoints[0]+ulpoints[1]
		n_p= len(ulpoints[0])
		fvs=[]
		for iu in range(1,n_p,1):
			il=iu+n_p
			fvs.append(np.array([il-1, il, iu]))
			fvs.append(np.array([il-1, iu, iu-1]))
		iu=n_p-1
		il = iu+n_p
		fvs.append(np.array([il, n_p, 0]))
		fvs.append(np.array([il, 0, iu]))
		return points,fvs
	def get_info(self) -> str:
		msg= super().get_info()
		msg="RSB Deck Hull"+msg
		return msg
class DBBBase(DBBBaseAll):
	def __init__(self, id, size:np.ndarray, position:np.ndarray,block_type:str,deck_hull:DBBDeckHull):
		super().__init__(id)
		self._position = position
		self._size = size
		self._type:str = block_type
		self._volume=0.0
		self._deck_hull:DBBDeckHull = deck_hull

	@property
	def deck_hull(self):
	    return self._deck_hull
	@property
	def type(self)->str:
	    return self._type
	@property
	def volume(self):
	    return self._volume

	def get_info(self) -> str:
		msg= super().get_info()
		msg += '\ntype = ' + str(self._type)
		msg += '\nvolume = ' + str(self.volume)
		return msg

	def regenerateMesh(self):
		self._genMesh()
		self._cutMesh(self.deck_hull.mesh)
		if self.is_closed():
			self._volume = self.calc_volume()
		else:
			self._volume=0.0

	def _genMesh(self):
		self.mesh = mf.make_block(block_dims=self.size, move_vector=self.position)

	def _cutMesh(self, cutting_mesh:om.TriMesh):  # za sada ako je position po y + a block dims po y neg nece radit
		if cutting_mesh.n_faces() > 0:
			if (self.size == 0.0).any():
				raise Exception("Block id: " + str(self.id) + " has one dimension as 0.")
			if use_gmsh_cut:
				cut_mesh = mf.cut_mesh(cutting_mesh, self.size, self.position)
			else:
				cut_mesh = mf.cut_mesh2(self.mesh, cutting_mesh, self.size, self.position)
			self.mesh = cut_mesh

	def calc_volume(self):
		return mf.calc_mesh_volume(self.mesh)

	def is_closed(self):
		bool = mf.is_mesh_closed(self.mesh)
		print("Mesh id " + str(self.id) + " is closed: " + "\n      " + str(bool) + "\n")
		return bool







class DBBCompartment(DBBBase):
	def __init__(self, id, size: np.ndarray, position: np.ndarray,block_type:str,deck_hull:DBBDeckHull,
				 zone:str,segment:DBBBase):
		super().__init__(id,size, position,block_type,deck_hull)
		self._zone = zone
		self._segment = segment
		self._segment.add_compartment(self)
		self.regenerateMesh()

	show_compartments = True

	@DBBBaseAll.show.getter
	def show(self):
	   return self._show and DBBCompartment.show_compartments

	@property
	def zone(self):
	    return self._zone

	@property
	def segment (self):
	    return self._segment

	def get_info(self) -> str:
		msg= super().get_info()
		msg="Compartment"+msg
		msg += '\nzone = ' + str(self.zone)
		msg += '\nsegment = ' + str(self.segment.id)
		return msg

class DBBCompartmentImportedMesh(DBBCompartment):
	def __init__(self, id, size: np.ndarray, position: np.ndarray,block_type:str,deck_hull:DBBDeckHull,
				 zone:str,segment:DBBBase,mesh_file_path,mesh_rot):
		self._mesh_file_path = mesh_file_path
		self._mesh_rot = mesh_rot
		super().__init__(id,size, position,block_type,deck_hull,zone,segment)


	def get_info(self) -> str:
		msg= super(DBBCompartment,self).get_info()
		msg="Predefined Compartment"+msg
		msg += '\nzone = ' + str(self.zone)
		msg += '\nsegment = ' + str(self.segment.id)
		return msg

	def regenerateMesh(self):
		self._genMesh()
	def _genMesh(self):

		mesh = om.read_trimesh(self._mesh_file_path)
		fvs = mesh.fv_indices()
		p = mesh.points()
		#r = R.from_euler('x', 90, degrees=True)
		listrot = self._mesh_rot.split(';')
		# rotate object to desired orientation
		if len(listrot) == 4:
			rot_order=listrot[0]
			rot_vals = float(listrot[1]),float(listrot[2]),float(listrot[3])
			r = R.from_euler(rot_order, rot_vals, degrees=True)
			rmat = r.as_matrix()
			p=np.dot(p,rmat.T)
		# determine coordinaes mins and maxes for shifting and scaling
		mins = p.min(axis=0)
		maxs = p.max(axis=0)
		d = maxs - mins
		#d = p.ptp(axis=0)
		o = (mins + maxs) * .5
		# move mesh object origin to the center of the object
		p=p-o
		scale = self.size/d
		scale = scale[0:2].min()
		q = p * scale
		mins = q.min(axis=0)
		#shift scaled mesh z origin to the bottom of the object
		q=q-np.array([0.0,0.0, mins[2]])
		#position the origin in the center of compartment with respect to x,y (z0 is on the deck level)
		co=self.position + np.array([self.size[0]/2,self.size[1]/2, 0.0])
		# move mesh to the final position
		q = q + co
		self.mesh =  om.TriMesh(q,fvs)

class DBBSegment(DBBBase):
	def __init__(self, id, size: np.ndarray, position: np.ndarray,block_type:str,deck_hull:DBBDeckHull,
				 zoneid: str, deck: DBBDeckBase):
		super().__init__(id, size, position,block_type,deck_hull)
		self._zoneid = zoneid
		self._compartments:Dict[str,DBBCompartment] = {}
		self._deck = deck
		self._deck.add_segment(self)
		self.regenerateMesh()

	show_segments = False

	@DBBBaseAll.show.getter
	def show(self):
	   return self._show and DBBSegment.show_segments

	@property
	def compartments(self) -> List[DBBCompartment]:
		return self._compartments.values()

	@property
	def zoneid(self):
		return self._zoneid

	@property
	def deck(self):
		return self._deck

	@property
	def num_compartments(self):
		return len(self._compartments)

	def add_compartment(self, compartment: DBBBase):
		self._compartments[compartment.id] = compartment

	def get_compartment(self, id):
		return self._compartments.get(id)

	def remove_compartment(self, id):
		return self._compartments.pop(id)

	def clear_compartments(self):
		self._compartments.clear()

	def get_info(self) -> str:
		msg = super().get_info()
		msg = "Segment" + msg
		msg += '\nzone = ' + str(self.zoneid)
		msg += '\ndeck = ' + str(self.deck.id)
		msg += '\nnum compartments = ' + str(self.num_compartments)
		return msg

class DBBDeckPlateHullSegment(DBBDeckBase):
	def __init__(self,deck_id, hullform:HullForm,  size: np.ndarray, position: np.ndarray,
				 rsb_hull_points:List[List[List[np.ndarray]]] = None,
				 rsb_deck_points:List[List[np.ndarray]] = None):
		super().__init__(deck_id, hullform, size, position)
		if rsb_hull_points != None:
			self._deckhull = DBBDeckHullRSB(self.id,self._hullform,self.size,self.position,self,rsb_hull_points)
		else:
			self._deckhull = DBBDeckHull(self.id,self._hullform,self.size,self.position,self)
		if rsb_deck_points != None:
			self._deckplate = DBBDeckPlateRSB(self.id, self._hullform, self.size, self.position,self,rsb_deck_points)
		else:
			self._deckplate = DBBDeckPlate(self.id, self._hullform, self.size, self.position,self)

		self.regenerateMesh()
		self._segments:Dict[str,DBBSegment] = {}

	@property
	def hull(self):
	    return self._deckhull
	@property
	def plate(self):
	    return self._deckplate

	@property
	def segments(self)->List[DBBSegment]:
	    return self._segments.values()

	@property
	def num_segments (self):
	    return len(self._segments)

	def add_segment(self,segment:DBBBase):
		self._segments[segment.id]=segment

	def get_segment(self,id):
		return self._segments.get(id)

	def remove_segment(self,id):
		return self._segments.pop(id)

	def clear_segments_compartments(self):
		for segment in self.segments:
			segment.clear_compartments()

	def append_all_compartments_list(self,all_compartments:List[DBBCompartment]):
		for segment in self.segments:
			for compartment in segment._compartments.values():
				all_compartments.append(compartment)

	def _genMesh(self):  # za keel koji je na 0 nema wl
		self.mesh = om.TriMesh()
		#self._gen_mesh_detail(self.plate.show,self.hull.show)

	def _gen_mesh_detail(self,doplate,dohull): # not used
		if dohull and doplate:
			self.mesh = DBBBaseAll.join_tria_meshes(self.hull.mesh,self.plate.mesh)
		elif dohull:
			self.mesh = DBBBaseAll.join_tria_meshes(self.hull.mesh,om.TriMesh())
		elif doplate:
			self.mesh = DBBBaseAll.join_tria_meshes(self.plate.mesh,om.TriMesh())
		else:
			self.mesh = om.TriMesh()



		
class DBBProblem_new():
	def __init__(self, fileName):
		self.filename = fileName
		self.hull:DBBHullFormAll = 0
		self._dictdecks: Dict[int,DBBDeckPlateHullSegment] = {}
		self._dictbulkheads: Dict[str, DBBBulkhead] = {}
		self._designs:Dict[str,str] = {}
		self.design_id=''
		if (fileName != ""):
			wbo_file,rsb_file,sb_file = self.read_problem()
			self.prepare_permanent_objects(wbo_file,rsb_file,sb_file)



	def prepare_permanent_objects(self, wbo_file:str, rsb_file:str, sb_file):
		self.prepare_decks(rsb_file, True)
		self.read_bulkheads(wbo_file,True)
		self.read_segments(sb_file, True)



	@property
	def decks(self)->List[DBBDeckPlateHullSegment]:
	    return self._dictdecks.values()
	@property
	def bulkheads(self)->List[DBBBulkhead]:
	    return self._dictbulkheads.values()
	@property
	def designs(self):
	    return self._designs

	@property
	def num_decks(self):
	    return len(self._dictdecks)

	def clear_current_design(self):
		for deck in self.decks:
			deck.clear_segments_compartments()


	def set_current_design(self,id_design, is_port, is_starb,do_port_strab_vis=True,dict_seg_show:Dict[str,bool]=None):
		if not id_design in self.designs:
			return [[],[],[]]
		comp_path = self.designs[id_design]

		allgeo = self.get_all_compartments()
		geo_add = []
		geo_reb = []
		geo_del = []
		geo_del = self.get_all_visible_geometries(allgeo)
		self.clear_current_design()
		self.read_compartments(comp_path,True)
		geo_add = self.get_all_compartments()
		if dict_seg_show is not None:
			for comp in geo_add:
				comp.set_show(dict_seg_show[comp.segment.id])
		if do_port_strab_vis:
			geo_add = self.set_geolist_portside_starboardside_visibility_(geo_add,is_port,is_starb)
		return geo_add, geo_reb, geo_del

	def prepare_decks(self,file_path,do_show):
		dict_rsb_data = {}
		if os.path.isfile(file_path):
			with open(file_path, "r") as csv_file:
				csv_reader = csv.DictReader(csv_file)
				#Type,Identifier,Deck,Vertices
				for row in csv_reader:  # each row contains 1 block data
					deck_code = int(row["Deck"])
					dict_rsb_data.setdefault(deck_code,[])
					deck_rsbs =dict_rsb_data[deck_code]
					rsb_data = []
					deck_rsbs.append(rsb_data)
					id = str(row["Identifier"])
					type = str(row["Type"])
					vertdata = str(row['Vertices'])
					rsb_data.append(id)
					rsb_data.append(type)
					rsb_data.append(vertdata)

		dict_rsb_deck_points = {}
		dict_rsb_hull_points = {}
		for deck_code,deck_rsbs in dict_rsb_data.items():
			deck_index=self.hull.deck_indexes[deck_code]
			deck_z = self.hull.get_deck_z_index(deck_index)
			lower_deck_code=deck_code-1
			lower_deck_index= self.hull.deck_indexes[lower_deck_code]
			lower_deck_z = self.hull.get_deck_z_index(lower_deck_index)
			dict_rsb_deck_points[deck_code]=[]
			dict_rsb_hull_points[lower_deck_code] = []
			for rsb_data in deck_rsbs:
				points = self._process_rsb_data(self.hull._hullform,[lower_deck_z,deck_z],rsb_data)
				dict_rsb_deck_points[deck_code].append(points[1])
				dict_rsb_hull_points[lower_deck_code].append(points)

		z_last = self.hull.get_deck_z_index(0) - self.hull.get_deck_z_index(1) + self.hull.get_deck_z_index(0)
		for key, value in self.hull.deck_indexes.items():
			z_deck = self.hull.get_deck_z_index(value)
			z_deck_up = z_last
			z_last = z_deck
			size = np.array([0, 0, z_deck_up - z_deck])
			position = np.array([0, 0, z_deck])
			deckhull = DBBDeckPlateHullSegment(key, self.hull, size, position,
											   dict_rsb_hull_points.get(key),dict_rsb_deck_points.get(key))
			self._dictdecks[key] = deckhull

	def read_bulkheads(self,file_path,do_show):
		if os.path.isfile(file_path):
			with open(file_path, "r") as csv_file:
				csv_reader = csv.DictReader(csv_file)
				#Type,Identifier,x,DeckStart,DeckEnd
				for row in csv_reader:  # each row contains 1 block data
					deck_code = int(row["DeckStart"])
					deck_start=self._dictdecks.get(deck_code)
					deck_code = int(row["DeckEnd"])
					deck_end = self._dictdecks.get(deck_code)
					if deck_start is not None:
						position = np.array([float(row["x"]),0.0,deck_start.position[2]])
						size = np.array([0.0,0.0,deck_end.position[2]-position[2]])
						id = str(row["Identifier"])
						type = str(row["Type"])
						bulkhead = DBBBulkhead(id,type,self.hull,size,position)
						self._dictbulkheads[id]=bulkhead

	def read_segments(self,file_path,do_show):
		with open(file_path, "r") as csv_file:
			csv_reader = csv.DictReader(csv_file)
			for row in csv_reader:  # each row contains 1 block data
				deck_code = int(row["Deck"])
				deck=self._dictdecks.get(deck_code)
				if deck is not None:
					position = np.array([float(row["Ax"]),float(row["Ay"]),deck.position[2]])
					size = np.array([float(row["b"]),float(row["a"]),deck.size[2]])
					id = str(row["Identifier"])
					type = str(row["Type"])
					segment = DBBSegment(id,size,position,type,deck.hull,'zone?',deck)
					segment.set_show(do_show)
					deck.add_segment(segment)
		self.add_fix_segments(do_show)
	def add_fix_segments(self,do_show):
		file_path = ""
		for value in self._designs.values():
			file_path=value
			break
		with open(file_path, "r") as csv_file:
			csv_reader = csv.DictReader(csv_file)
			for row in csv_reader:  # each row contains 1 block data
				deck_code = int(row["Deck"])
				deck=self._dictdecks.get(deck_code)
				if deck is not None:
					id_seg = str(row["Segment"])
					if id_seg == 'NO':
						position = np.array([float(row["Ax"]), float(row["Ay"]), deck.position[2]])
						size = np.array([float(row["b"]), float(row["a"]), deck.size[2]])
						id = str(row["Identifier"])
						type = str(row["Type"])
						zone = str(row["Zone"])
						segment = DBBSegment('NO'+id, size, position, type, deck.hull, zone, deck)
						segment.set_show(do_show)
						deck.add_segment(segment)
						comp = DBBCompartment(id, size, position, type, deck.hull, zone, segment)
						comp.set_show(do_show)
						segment.add_compartment(comp)

	def read_compartments(self,file_path,do_show):
		with open(file_path, "r") as csv_file:
			csv_reader = csv.DictReader(csv_file)
			for row in csv_reader:  # each row contains 1 block data
				deck_code = int(row["Deck"])
				deck=self._dictdecks.get(deck_code)
				if deck is not None:
					id_seg = str(row["Segment"])
					id = str(row["Identifier"])
					if id_seg != 'NO':
						id_seg = 'S' + id_seg
					else:
						id_seg = 'NO'+id
					segment = deck.get_segment(id_seg)
					if segment is not None:
						type = str(row["Type"])
						zone = str(row["Zone"])
						position = np.array([float(row["Ax"]), float(row["Ay"]), deck.position[2]])
						size = np.array([float(row["b"]), float(row["a"]), deck.size[2]])
						if type == "SMC":
							smc_path = self._abspath+str(row["add_info1"])
							smc_rot_data = str(row["add_info2"])
							comp = DBBCompartmentImportedMesh(id, size, position, type, deck.hull, zone, segment,
															  smc_path,smc_rot_data)
						else:
							comp = DBBCompartment(id,size,position,type,deck.hull,zone,segment)
						comp.set_show(do_show and segment._show)
						segment.add_compartment(comp)
					else:
						print("Error, segment should exist!")


	def prepare_exported_data(self, folder_path, sf_char):
		self._designs.clear()
		wb_file=""
		rsb_file = ""
		sb_file = ""
		if os.path.isdir(folder_path):
			_, _, filenames = next(os.walk(folder_path))
			for file in filenames:
				i1_CB = file.find('_CB_exp')
				if i1_CB > -1:
					key = file[:i1_CB]
					self._designs[key]=folder_path+sf_char+file
				else:
					i1_WB = file.find('WB_exp')
					i1_RSB = file.find('RSB_exp')
					i1_SB = file.find('SB_exp')
					if i1_WB > -1:
						wb_file = folder_path+sf_char+file
					elif i1_RSB > -1:
						rsb_file = folder_path+sf_char+file
					elif i1_SB > -1:
						sb_file = folder_path+sf_char+file

				pass
			pass
		return wb_file,rsb_file,sb_file
	def set_portside_starboardside_visibility(self, show_portside, show_starbside,do_force_add=False)->Tuple[List[DBBBaseAll]]:
		allgeo = self.get_all_geometries()
		for geo in allgeo:
			geo.regenerate_mesh_port_starboard(show_portside,show_starbside)
		geo_add =[]
		geo_reb = []
		geo_del = []
		if do_force_add:
			geo_add = self.get_all_visible_geometries(allgeo)
		else:
			geo_reb = self.get_all_visible_geometries(allgeo)
		return geo_add,geo_reb,geo_del

	def set_geolist_portside_starboardside_visibility_(self,geolist:List[DBBBaseAll], show_portside, show_starbside)->List[DBBBaseAll]:
		for geo in geolist:
			geo.regenerate_mesh_port_starboard(show_portside,show_starbside)
		return self.get_all_visible_geometries(geolist)

	def get_changes_to_previous_state(self,ref_allgeo)->Tuple[List[DBBBaseAll]]:
		allgeo = self.get_all_visible_geometries()
		geo_add = []
		geo_reb = []
		geo_del = []
		for geo in ref_allgeo:
			if geo not in allgeo:
				geo_del.append(geo)

		for geo in allgeo:
			if geo not in ref_allgeo:
				geo_add.append(geo)

		return geo_add, geo_reb, geo_del


	def get_all_visible_geometries(self,all_geo:List[DBBBaseAll]=None)->List[DBBBaseAll]:
		show_geo = []
		if all_geo is None:
			all_geo = self.get_all_geometries()
		for	geo in all_geo:
			if geo.show:
				show_geo.append(geo)
		return show_geo

	def get_all_geometries(self)->List[DBBBaseAll]:
		geometires = []
		for bulk in self.bulkheads:
			geometires.append(bulk)
		for deck in self.decks:
			if deck.plate is not None:
				geometires.append(deck.plate)
			if deck.hull is not None:
				geometires.append(deck.hull)
			if deck.segments is not None:
				for segment in deck.segments:
					geometires.append(segment)
					if segment is not None:
						for compartment in segment.compartments:
							geometires.append(compartment)
		return geometires

	def get_all_compartments(self)->List[DBBCompartment]:
		geometires = []
		for deck in self.decks:
			if deck.segments is not None:
				for segment in deck.segments:
					if segment is not None:
						for compartment in segment.compartments:
							geometires.append(compartment)
		return geometires

	def read_problem(self):
		with open(self.filename, "r") as csv_file:
			csv_reader = csv.DictReader(csv_file)
			for row in csv_reader:	#each row contains 1 block data
				hull_form_input = str(row["hull_form_file"])
				dbb_designs = str(row["dbb_designs_folder"])

			abspath1 = '\\'.join(self.filename.split('\\')[0:-1])
			abspath2 = '/'.join(self.filename.split('/')[0:-1])
			if len(abspath2) > len(abspath1):
				sf_char = '/'
				self._abspath = abspath2 + '/'
				hull_form_input = hull_form_input.replace('\\','/')
				dbb_designs = dbb_designs.replace('\\','/')
			else:
				sf_char='\\'
				self._abspath = abspath1 + '\\'
				hull_form_input = hull_form_input.replace('/','\\')
				dbb_designs = dbb_designs.replace( '/','\\')
			hull_form_input = self._abspath+hull_form_input
			dbb_designs = self._abspath + dbb_designs
			wb_file,rsb_file, sb_file = self.prepare_exported_data(dbb_designs, sf_char)
			self.hull = DBBHullFormAll(hull_form_input)
			return wb_file,rsb_file,sb_file

	def _process_rsb_data(self,hullform:HullForm,z_dck:List[float], rsb_data: List[str]):
		verts = rsb_data[2].strip()
		splited = verts.split(" ")
		points = []
		for z in z_dck:
			points.append([])
		for i in range(0, len(splited), 2):
			deck_points = None
			xmin = xmax = 0
			wlines = None
			if splited[i].lower().startswith("p"):
				wlines = hullform.wlinesPos
			elif splited[i].lower().startswith("s"):
				wlines = hullform.wlinesNeg
			else:
				x = float(splited[i])
				y = float(splited[i + 1])
				for i_z in range(len(z_dck)):
					points[i_z].append(np.array([x, y, z_dck[i_z]]))

			if wlines is not None:
				deck_points = []
				for z in z_dck:
					xmin = float(splited[i][1:])
					xmax = float(splited[i + 1][1:])
					for wline in wlines:
						if np.isclose(z, wline[0][2]):
							deck_points.append((np.asarray(wline)).tolist())
				tmp = -9999
				if xmin > xmax:
					tmp = xmin
					xmin = xmax
					xmax = tmp
				for i_z in range(len(z_dck)):
					deck_points[i_z] = self.trim_sorted_points_by_x_min_max(xmin, xmax, deck_points[i_z])
					if tmp > -9999:
						deck_points[i_z].reverse()
					for p in deck_points[i_z]:
						points[i_z].append(p)
		return points

	def trim_sorted_points_by_x_min_max(self, xmin, xmax, points):
		trim_points = []
		n_p = len(points)
		x_tol = 0.0001
		i = 0
		while (i < n_p and points[i][0] < xmin):
			i += 1
		if i > 0 and (points[i][0] - x_tol) > xmin:
			p = np.array([xmin, 0.0, points[i][2]])
			p[1] = points[i - 1][1] + (points[i][1] - points[i - 1][1]) / (points[i][0] - points[i - 1][0]) * (
						xmin - points[i - 1][0])
			trim_points.append(p)
		while (i < n_p and points[i][0] <= xmax):
			trim_points.append(points[i])
			i += 1
		if i > 0 and i < n_p and (points[i - 1][0] + x_tol) < xmax:
			p = np.array([xmax, 0.0, points[i][2]])
			p[1] = points[i - 1][1] + (points[i][1] - points[i - 1][1]) / (points[i][0] - points[i - 1][0]) * (
						xmax - points[i - 1][0])
			trim_points.append(p)

		return trim_points

		
		