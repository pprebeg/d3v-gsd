import openmesh as om
import os
import numpy as np
import copy
import csv
import dbbdir.myfunctions as mf
from hullformdir.hullform import HullForm
#d3v imports
from extendedgeometry import ExtendedGeometry
from typing import List, Dict,Tuple

class DBBProblem ():
	def __init__(self,fileName):
		self.filename=fileName
		self.hull = 0
		self.decks = []
		self.dbbs = []


		if (fileName != ""):
			self.readProblem()


	def readProblem(self):
		with open(self.filename, newline='') as csvfile:
			hfr = csv.reader(csvfile, delimiter='\t', quotechar='|')
			data = []
			for row in hfr:
				rown = []
				for x in row:
					rown.append(x)
				data.append(rown)


		abspath1 = '\\'.join(self.filename.split('\\')[0:-1])
		abspath2 = '/'.join(self.filename.split('/')[0:-1])
		if len(abspath2) > len(abspath1):
			abspath = abspath2 + '/'
		else:
			abspath = abspath1 + '\\'
		
		huf_path = abspath + data[0][1]
		block_data_path = abspath + data[1][1]
		#make hull from huf file:
		self.hull= DBBHullForm(huf_path)
		deckz_list= self.hull.pdecks[:-1] #bez keela
		#deckz_list = []
		#deckz_list.append(self.hull.pdecks[2])
		#deckz_list.append(self.hull.pdecks[3])
		#deckz_list.append(self.hull.pdecks[4])
		num_decks = len(self.hull.pdecks)-1
		for key,value in self.hull.dict_decks.items():
			if value < (num_decks):
				self.decks.append(DBBDeck(self.hull, key))
			
		#make blocks
		with open(block_data_path, "r") as csv_file:
			csv_reader = csv.DictReader(csv_file)
			for row in csv_reader:	#each row contains 1 block data
				deck  = int(row["deck"])
				segment = str(row["segment"])
				zone = str(row["zone"])
				Ax = float(row["Ax"])
				Ay = float(row["Ay"])	
				Az = self.decks[deck].z
				x = float(row["b"])
				y = float(row["a"])
				id = str(row["identifier"])
				type = str(row["type"])
				try:
					#z = self.decks[deck - 1].z - self.decks[deck].z
					index_deck = self.hull.dict_decks[deck]
					z= self.decks[index_deck-1].z-self.decks[index_deck].z
				except IndexError:
					print("Invalid deck position")
				else:
					block_dims = np.array([x,y,z])
					position_A = np.array([Ax,Ay])
				
					block = DBB(self.hull, self.decks[index_deck], block_dims, position_A, abspath, id, type,segment,zone)
					self.dbbs.append(block)
		self.segment_blocks = self.prep_dbb_organize_dicts()
		#print(self.hull.wlinesNeg)
		#print(self.hull.wlinesPos[1])
		#print(self.hull)
		#print(len(self.decks))
		#print(self.dbbs)
		#print(self.filename)
	def prep_dbb_organize_dicts(self):
		d_segments = {}
		for dbb in self.dbbs:
			key = dbb.segment
			dbblist = d_segments.get(key)
			if dbblist is None:
				dbblist = []
				d_segments[key]= dbblist
			dbblist.append(dbb)
		return d_segments



		
		
		
class DBBHullForm (HullForm):
	def __init__(self, fileName, scale = 1):
		super().__init__(fileName)		#vec u inicijalizaciji stvara formu
		self.position = np.array([0.,0.,0.])
		self.centroid = np.array([0.,0.,0.])
		self.LOA = self.shipdata["loa_val"]
		self.centroid = np.array([self.LOA / 2, 0, 0])  # sredina samo za x za sada
		#self.mesh = mf.hard_merge_meshes([self.mesh])

	#def readHullForm(self):
		#HullForm.__init__()


	def regenerateMesh(self):
		self.mesh = mf.make_form(scale = self.scale, move_vector = self.position)

	def move(self, move_vector):
		self.position += move_vector
		self.mesh = mf.move_mesh(self.mesh, move_vector)
		#self.regenerateMesh()

	def setPosition(self, new_position):
		old_position = self.position
		self.position = new_position
		self.mesh = mf.move_mesh(self.mesh, new_position - old_position)
		
	def testMesh(self, scale):
		self.scale = scale
		self.mesh = mf.make_form(scale = self.scale, move_vector = self.position)

		
class DBBDeck(ExtendedGeometry):
	def __init__(self, hullform, deck_code):
		super().__init__()
		self.hullform = hullform
		self._id = deck_code
		self.genMesh()

	@property
	def id(self):
	    return self._id

	@property
	def deckIndex(self):
	    return self.hullform.dict_decks[self._id]

	@property
	def z(self):
	    return self.hullform.pdecks[self.deckIndex]

	def regenerateMesh(self):
		self.mesh = om.TriMesh()

	def genMesh(self):	#za keel koji je na 0 nema wl
		for wline in self.hullform.wlinesPos:
			if np.isclose(self.z, wline[0][2]):
				deck_points = np.asarray(wline)
				break
		self.mesh = mf.make_deck(deck_points, subdivide = False)





	def move(self, move_vector):
		self.z += move_vector[2]
		#self.regenerateMesh()
		self.mesh = mf.move_mesh(self.mesh, move_vector)
		
		#self.position[0] = self.position[0] + dx
		#self.position[1] = self.position[0] + dy
		#self.position[2] = self.position[0] + dz
		#self.regenerateMesh()

	def setPosition(self, new_position):
		old_position = self.position
		self.position = new_position
		self.mesh = mf.move_mesh(self.mesh, new_position - old_position)




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
		is_changed = (do_show==self._show)
		self._show = do_show
		return is_changed

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
		npts1 = points1.size
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
		msg += '\ndimensions = ' + str(self.block_dims)
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
		if not self._do_portside:
			self.mesh = DBBBaseAll.cut_mesh_with_plane(self.mesh, -1, False, i_test_dims, tol, lbs)
		elif not self._do_starbside:
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
		self._loa = self._hullform.shipdata["loa_val"]
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
	    return self._loa

	def _genMesh(self):
		self.mesh = self._hullform.mesh

	def move(self, move_vector):
		self.position += move_vector
		self.mesh = mf.move_mesh(self.mesh, move_vector)

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


class DBBDeckHull(DBBDeckUnitBase):
	def __init__(self, deck_id, hullform, size: np.ndarray, position: np.ndarray, parent_deck_segment: DBBBaseAll):
		super().__init__(deck_id, hullform, size, position,parent_deck_segment)
		self.regenerateMesh()

	def _genMesh(self):  # za keel koji je na 0 nema wl
		zlb = self.position[2]
		zub = zlb+self.size[2]
		lbs=[zlb]
		ubs =[zub]
		i_test_dims = [2]
		self.mesh = DBBBaseAll.cut_mesh_with_plane(self._hullform.mesh,-1,False,i_test_dims,0.0,lbs,ubs)

class DBBBase(DBBBaseAll):
	def __init__(self, id, size:np.ndarray, position:np.ndarray,block_type:str):
		super().__init__(id)
		self._position = position
		self._dimensions = size
		self._type:str = block_type
		self._volume=0.0

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
		self._cutMesh()
		if self.is_closed():
			self._volume = self.calc_volume()
		else:
			self._volume=0.0

	def _genMesh(self):
		self.mesh = mf.make_block(block_dims=self.size, move_vector=self.position)

	def _cutMesh(self, cutting_mesh:om.TriMesh):  # za sada ako je position po y + a block dims po y neg nece radit
		cut_mesh = mf.cut_mesh2(self.mesh, cutting_mesh, self.size, self.position)
		# if cut_mesh is not None:
		self.mesh = cut_mesh
		pass

	def calc_volume(self):
		return mf.calc_mesh_volume(self.mesh)

	def is_closed(self):
		 return mf.is_mesh_closed(self.mesh)







class DBBCompartment(DBBBase):
	def __init__(self, id, size: np.ndarray, position: np.ndarray,block_type:str,zone:str,segment:DBBBase):
		super().__init__(id,size, position,block_type)
		self._zone = zone
		self._segment = segment
		self._segment.add_compartment(self)

	@property
	def zone(self):
	    return self._zone

	@property
	def segment (self):
	    return self._segment

	def get_info(self) -> str:
		msg= super().get_info()
		msg += '\nzone = ' + str(self.zone)
		msg += '\nsegment = ' + str(self.segment.id)
		return msg


class DBBSegment(DBBBase):
	def __init__(self, id, size: np.ndarray, position: np.ndarray,block_type:str, zoneid: str, deck: DBBDeckBase):
		super().__init__(id, size, position,block_type)
		self._zoneid = zoneid
		self._compartments:Dict[str,DBBCompartment] = {}
		self._deck = deck
		self._deck.add_segment(self)

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
		msg += '\nzone = ' + str(self.zoneid)
		msg += '\ndeck = ' + str(self.deck.id)
		msg += '\nnum compartments = ' + str(self.num_compartments)
		return msg

class DBBDeckPlateHullSegment(DBBDeckBase):
	def __init__(self,deck_id, hullform:HullForm,  size: np.ndarray, position: np.ndarray):
		super().__init__(deck_id, hullform, size, position)
		self._deckhull = DBBDeckHull(self.id,self._hullform,self.size,self.position,self)
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

	def clear_segments(self):
		for segment in self.segments:
			segment.clear_compartments()
		self._segments.clear()

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
		
class DBB(ExtendedGeometry):
	def __init__(self, hullform, deck:DBBDeck,block_dims, position, abspath, id, type,segment,zone):
		super().__init__()
		self.folderpath = abspath
		self.hullform= hullform
		self.deck=deck
		self.position = np.array([position[0],position[1],deck.z])
		self.block_dims=block_dims
		self.id = id
		self.type = type
		self.segment = segment
		self.zone = zone
		self.genMesh()
		self.cutMesh()
		#print(self.hullform.filename)
		self.volume = mf.calc_mesh_volume(self.mesh)
	def get_info(self)->str:
		msg = 'DBB'
		msg += '\nid = '+str(self.id)
		msg += '\ntype = ' + str(self.type)
		msg += '\nsegment = ' + str(self.segment)
		msg += '\nzone = ' + str(self.zone)
		msg += '\nposition = ' + str(self.position)
		msg += '\ndimensions = ' + str(self.block_dims)
		msg += '\nvolume = ' + str(self.volume)
		return msg
	def regenerateMesh(self):
		self.mesh= mf.make_block(block_dims = self.block_dims, move_vector = self.position)

	def move(self, move_vector):
		self.position += move_vector
		#self.regenerateMesh()
		self.mesh = mf.move_mesh(self.mesh, move_vector)
		
		#self.position[0] = self.position[0] + dx
		#self.position[1] = self.position[0] + dy
		#self.position[2] = self.position[0] + dz
		#self.regenerateMesh()

	def setPosition(self, new_position):

		self.position = new_position
		self.genMesh()

		old_position = self.position
		self.position = new_position
		self.mesh = mf.move_mesh(self.mesh, new_position - old_position)
		
	def genMesh(self):
		print(self.id)
		#self.mesh = mf.make_block_from_unit_csv(self.block_dims, self.position, self.folderpath)
		# y_block_min = self.position[1]
		# y_block_max = self.position[1] + self.block_dims[1]
		# if (np.sign(y_block_min) != np.sign(y_block_max)) and (y_block_min != 0.0) and (y_block_max != 0.0):		#if ymin and ymax have different signs and are not on x axis(0)
			# self.mesh = mf.make_split_block(block_dims = self.block_dims, move_vector = self.position)
		# else:
		self.mesh= mf.make_block(block_dims = self.block_dims, move_vector = self.position)

	def cutMesh(self):			#za sada ako je position po y + a block dims po y neg nece radit
		# print("cutting mesh")
		# y_block_min = self.position[1]
		# y_block_max = self.position[1] + self.block_dims[1]
		# if (np.sign(y_block_min) != np.sign(y_block_max)) and (y_block_min != 0.0) and (y_block_max != 0.0):		#if ymin and ymax have different signs and are not on x axis(0)
			# print("splitting")
			# data = mf.make_split_block(block_dims = self.block_dims, move_vector = self.position)
			# pos1 = self.position
			# bdims1 = copy.copy(self.block_dims)
			# bdims1[1] = -pos1[1]
			
			# cut_mesh1 = mf.cut_mesh(data[0], self.hullform.mesh, bdims1, pos1) 
			# pos2 = copy.copy(self.position)
			# pos2[1] = 0.0
			# bdims2 = copy.copy(self.block_dims)
			# bdims2[1] = self.block_dims[1] + self.position[1]  #pos[1] je negativan
			# cut_mesh2 = mf.cut_mesh(data[1], self.hullform.mesh, bdims2, pos2)		#different block dims and pos from 1
			#cut_mesh = cut_mesh2
			# cut_mesh = mf.soft_merge_meshes([cut_mesh1, cut_mesh2])
		# else:
		cut_mesh = mf.cut_mesh2(self.mesh, self.hullform.mesh, self.block_dims, self.position)
		
		# if cut_mesh is not None:
		self.mesh = cut_mesh
		
		
		
		
		#mf.fit_block_to_form(self.mesh, self.block_dims, self.position, self.hullform.mesh)
		#mf.cut_meshes(self.mesh, self.hullform.mesh)
		pass
		
	def calcVolume(self):
		print("\nSelected mesh volume:\n")
		print(mf.calc_mesh_volume(self.mesh))
		
	def IsClosed(self):
		print(mf.is_mesh_closed(self.mesh))
		
		
		
		
class DBBProblem_new():
	def __init__(self, fileName):
		self.filename = fileName
		self.hull:DBBHullFormAll = 0
		self._dictdecks: Dict[int,DBBDeckPlateHullSegment] = {}
		self._designs:Dict[str,List[str]] = {}
		self.design_id=''
		if (fileName != ""):
			self.read_problem()
	@property
	def decks(self):
	    return self._dictdecks.values()
	@property
	def designs(self):
	    return self._designs

	@property
	def num_decks(self):
	    return len(self._dictdecks)

	def clear_current_design(self):
		for deck in self.decks:
			deck.clear_segments()
		self._dictdecks.clear()

	def set_current_design(self,id_design):
		id_design='Ind_10'
		if not id_design in self.designs:
			return [[],[],[]]
		[seg_path, comp_path] = self.designs[id_design]

		allgeo = self.get_all_geometries()
		geo_add = []
		geo_reb = []
		geo_del = self.get_all_visible_geometries(allgeo)
		self.clear_current_design()
		z_last=self.hull.get_deck_z_index(0)-self.hull.get_deck_z_index(1)+self.hull.get_deck_z_index(0)
		for key, value in self.hull.deck_indexes.items():
			z_deck = self.hull.get_deck_z_index(value)
			z_deck_up = z_last
			size = np.array([0, 0, z_deck_up-z_deck])
			position = np.array([0, 0,z_deck])
			self._dictdecks[key] = DBBDeckPlateHullSegment(key,self.hull,size,position)

		self.read_segments(seg_path)
		self.read_compartments(comp_path)
		allgeo = self.get_all_geometries()
		geo_add = self.get_all_visible_geometries(allgeo)



		return geo_add, geo_reb, geo_del
	def read_segments(self,file_path):
		with open(file_path, "r") as csv_file:
			csv_reader = csv.DictReader(csv_file)
			for row in csv_reader:  # each row contains 1 block data
				deck_code = int(row["Deck"])
				deck=self._dictdecks.get(deck_code)
				if deck is not None:
					size = np.array([float(row["Ax"]),float(row["Ay"]),deck.position[2]])
					position = np.array([float(row["b"]),float(row["a"]),deck.size[2]])
					id = str(row["Identifier"])
					type = str(row["Type"])
					segment = DBBSegment(id,size,position,type,'zone?',deck)
					deck.add_segment(segment)

	def read_compartments(self,file_path):
		with open(file_path, "r") as csv_file:
			csv_reader = csv.DictReader(csv_file)
			for row in csv_reader:  # each row contains 1 block data
				deck_code = int(row["Deck"])
				deck=self._dictdecks.get(deck_code)
				if deck is not None:
					id_seg = str(row["Segment"])
					segment = deck.get_segment(id_seg)
					if segment is not None:
						size = np.array([float(row["Ax"]),float(row["Ay"]),deck.position[2]])
						position = np.array([float(row["b"]),float(row["a"]),deck.size[2]])
						id = str(row["Identifier"])
						type = str(row["Type"])
						zone = str(row["Zone"])
						comp = DBBCompartment(id,size,position,type,zone,segment)
						segment.add_compartment(comp)


	def set_design_list(self,folder_path,sf_char):
		self._designs.clear()
		if os.path.isdir(folder_path):
			_, _, filenames = next(os.walk(folder_path))
			for file in filenames:
				i1_CB = file.find('_CB_exp')
				i1_SB = file.find('_SB_exp')
				i1 = max(i1_CB,i1_SB)
				if i1 > -1:
					key = file[:i1]
					list = self._designs.get(key,['',''])
					self._designs[key]=list
					file_path = folder_path+sf_char+file
					if i1_CB > -1:
						list[1]=file_path
					elif i1_SB > -1:
						list[0]=file_path
				pass
			pass

	def set_portside_starboardside_visibility(self, show_portside, show_starbside)->Tuple[List[DBBBaseAll]]:
		allgeo = self.get_all_geometries()
		for geo in allgeo:
			geo.regenerate_mesh_port_starboard(show_portside,show_starbside)
		geo_add =[]
		geo_reb = self.get_all_visible_geometries(allgeo)
		geo_del = []
		return geo_add,geo_reb,geo_del

	def set_deck_visibility(self, show_deck_plate, show_deck_hull,d_all_plate,d_all_hull)->Tuple[List[DBBBaseAll]]:
		geo_reb = []
		for deck in self.decks:
			is_changed_p = deck.plate.set_show(show_deck_plate and d_all_plate[deck.id])
			is_changed_h = deck.hull.set_show(show_deck_hull and d_all_hull[deck.id])
			if (is_changed_p or is_changed_h):
				deck.regenerateMesh()
				deck.post_gen_mesh()
				if deck.show:
					geo_reb.append(deck)

		geo_add =[]
		geo_del = []
		return geo_add,geo_reb,geo_del

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
				abspath = abspath2 + '/'
				hull_form_input = hull_form_input.replace('\\','/')
				dbb_designs = dbb_designs.replace('\\','/')
			else:
				sf_char='\\'
				abspath = abspath1 + '\\'
				hull_form_input = hull_form_input.replace('/','\\')
				dbb_designs = dbb_designs.replace( '/','\\')
			hull_form_input = abspath+hull_form_input
			dbb_designs = abspath + dbb_designs
			self.set_design_list(dbb_designs,sf_char)
			self.hull = DBBHullFormAll(hull_form_input)



	def readProblem(self):
		with open(self.filename, newline='') as csvfile:
			hfr = csv.reader(csvfile, delimiter='\t', quotechar='|')
			data = []
			for row in hfr:
				rown = []
				for x in row:
					rown.append(x)
				data.append(rown)

		abspath1 = '\\'.join(self.filename.split('\\')[0:-1])
		abspath2 = '/'.join(self.filename.split('/')[0:-1])
		if len(abspath2) > len(abspath1):
			abspath = abspath2 + '/'
		else:
			abspath = abspath1 + '\\'

		huf_path = abspath + data[0][1]
		block_data_path = abspath + data[1][1]
		# make hull from huf file:
		self.hull = DBBHullForm(huf_path)
		deckz_list = self.hull.pdecks[:-1]  # bez keela
		# deckz_list = []
		# deckz_list.append(self.hull.pdecks[2])
		# deckz_list.append(self.hull.pdecks[3])
		# deckz_list.append(self.hull.pdecks[4])
		num_decks = len(self.hull.pdecks) - 1
		for key, value in self.hull.dict_decks.items():
			if value < (num_decks):
				self.decks.append(DBBDeck(self.hull, key))

		# make blocks
		with open(block_data_path, "r") as csv_file:
			csv_reader = csv.DictReader(csv_file)
			for row in csv_reader:  # each row contains 1 block data
				deck = int(row["deck"])
				segment = str(row["segment"])
				zone = str(row["zone"])
				Ax = float(row["Ax"])
				Ay = float(row["Ay"])
				Az = self.decks[deck].z
				x = float(row["b"])
				y = float(row["a"])
				id = str(row["identifier"])
				type = str(row["type"])
				try:
					# z = self.decks[deck - 1].z - self.decks[deck].z
					index_deck = self.hull.dict_decks[deck]
					z = self.decks[index_deck - 1].z - self.decks[index_deck].z
				except IndexError:
					print("Invalid deck position")
				else:
					block_dims = np.array([x, y, z])
					position_A = np.array([Ax, Ay])

					block = DBB(self.hull, self.decks[index_deck], block_dims, position_A, abspath, id, type, segment,
								zone)
					self.dbbs.append(block)
		self.segment_blocks = self.prep_dbb_organize_dicts()

	# print(self.hull.wlinesNeg)
	# print(self.hull.wlinesPos[1])
	# print(self.hull)
	# print(len(self.decks))
	# print(self.dbbs)
	# print(self.filename)
	def prep_dbb_organize_dicts(self):
		d_segments = {}
		for dbb in self.dbbs:
			key = dbb.segment
			dbblist = d_segments.get(key)
			if dbblist is None:
				dbblist = []
				d_segments[key] = dbblist
			dbblist.append(dbb)
		return d_segments
		
		