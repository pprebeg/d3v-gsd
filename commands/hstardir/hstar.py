from extendedgeometry import ExtendedGeometry
import openmesh as om
import os
import numpy as np
import math as Math
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
		#self.mesh = self._hullform.mesh
		self.mesh=self.generateHSMSH()

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

	@property
	def wlinesPos(self):
		return self._hullform.wlinesPos
	@property
	def wlinesNeg(self):
		return self._hullform.wlinesNeg

	@property
	def wlKeel(self):
		return self._hullform.wlKeel
	@property
	def slope(self):
		return self._hullform.shipdata["sa_val"]
	@property
	def slope(self):
		return self._hullform.shipdata["sa_val"]

	@property
	def transomTip(self):
		transomTip=self._hullform.shipdata["draft_val"] * self._hullform.shipdata["td_val"]
		return transomTip

	@property
	def x0(self):
		x0=self.D*(1-self._hullform.shipdata["td_val"])*Math.tan(self.slope)
		return x0


	def _genMesh(self):
		pass

	#Hstar geometry
	def getNewPointsonWL(self):
		wlinesPos=self.wlinesPos
		wlKeel=self.wlKeel
		loa=self.LOA
		#slope = self.slope
		draft = self.D
		transomTip = self.transomTip
		#deck = wlinesPos[0][0][2]
		xPos=self.getXpositions()
		#x0 = self.x0
		wlinesPos=self.clearWLframes(wlinesPos)
		#nfr = 20
		#nsec = nfr - 1
		#r = loa / nfr
		frames=[]
		#wlframes = []
		cla = []
		clf = []
		for i in range(len(xPos)-1):
			frames.append([])
		for wl in wlinesPos:
			#wlframe = []
			#xfr = r
			k=0
			xfr=xPos[k]
			#r0 = 1
			#z = 0
			if wl[0][2]<=draft:
				cla.append(wl[0])
				clf.append(wl[-1])
			#b=False
			"""if wl[0][2] >= transomTip:  # problem R0, x=x_transomTip
				xfr = x0
				r0 = 0
				b = True
			if wl[0][0] > xfr or wl[1][0] > xfr:  # rijiešen problem vl 12-23
				a = wl[0][0] // 5
				xfr = (a + 1) * r
				r0 = 0"""
			for i in range(0, len(wl) - 1):
				j = i
				p1 = wl[j]
				p2 = wl[j + 1]
				zp = p1[2]
				while p1[0]>xfr:
					k+=1
					xfr=xPos[k]

				while p2[0]>xfr:
					if p1[0]<=xfr:
						yp = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xfr - p1[0])
						if np.isclose(zp,transomTip):
							pointType=1
						else:
							pointType=0
						ip = [yp, zp,pointType]
						#wlframe.append(ip)
						frames[k].append(ip)
						k+=1
						xfr=xPos[k]


				"""if p1[0] <= xfr and p2[0] > xfr:
					yp = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xfr - p1[0])
					ip = [xfr, yp, zp]
					wlframe.append(ip)
					#xfr += r
					k=+1
					xfr=xPos[k]


					if b and zp < deck and zp > transomTip and z == 0:
						xfr = xfr - x0
						z = 1"""

			#wlframes.append(wlframe)
		#xfram=frames[0][0][0]
		#b=[]

		cla.reverse()
		wlKeel = cla + wlKeel + clf
		#xfr = x0
		wlframeK = []
		k=0
		xfr=xPos[k]
		for ii in range(len(wlKeel) - 1):
			#if len(wlframeK) == 1:
				#xfr = r
			#xfr=xPos[k]
			p1 = wlKeel[ii]
			p2 = wlKeel[ii + 1]
			while p2[0]>xfr:
				if p1[0]<=xfr:
					yp = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xfr - p1[0])
					zp = p1[2] + (p2[2] - p1[2]) / (p2[0] - p1[0]) * (xfr - p1[0])
					ip = [xfr, yp, zp]
					wlframeK.append(ip)
					k+=1
					xfr=xPos[k]

		k=0
		for frame in frames:
			#b.append(frame)
			#xfram=frame[1][0]

			zp=wlframeK[k][2]
			if np.isclose(zp, transomTip):
				pointType = 1
			else:
				pointType = 0
			#ip = [yp, zp, pointType]
			point=[wlframeK[k][1],wlframeK[k][2],pointType]
			#point.
			frame.insert(0,point)
			fr = [xPos[k], len(frame)]
			frame.insert(0,fr)
			k+=1


		#wlframes.insert(0,wlframeK)
		#wlframes=self.clearWLframes(wlframes)

		return frames

	def getInstersectionPoint(self,p1,p2,x):
		zp = p1[2]
		yp = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (x - p1[0])
		ip = [x, yp, zp]
		return ip

	def getXpositions(self):
		x0=self.x0
		x1=self.wlKeel[0][0]
		xmid=self.LOA/2
		x2=self.wlKeel[-1][0]
		xend=self.LOA
		l1=self.getXposition(x0,x1,1)
		l2=self.getXposition(x1,xmid,5)
		l3=self.getXposition(xmid,x2,5)
		l4=self.getXposition(x2,xend+0.5,0.5)
		xPos=l1+l2+l3+l4
		return xPos

	def getXposition(self,x1,x2,step:int):
		l=np.arange(x1,x2,step)
		l=l.tolist()
		return l

	def getAdditionalFrames(self, wlinesPos, wlframes, clf):

		for wl in wlinesPos:
			print()
		pass

	def clearWLframes(self, wlframes):
		draft = self.D
		i = 0
		vl = []
		for wl in wlframes:
			z = wl[0][2]
			if z <= draft:
				vl.append(wl)
		wlframes = vl
		return wlframes


	def getFrames(self,n):
		ind = []
		# ind.append(x0)
		slope = self.slope
		draft = self.D
		transomTip = self.transomTip
		loa=self.LOA
		wlframes = self.getNewPointsonWL(loa, self.wlinesPos, self.wlKeel)
		xPos=self.getXpositions()
		# deck = wlinesPos[0][0][2]
		x0 = self.x0
		r = loa / n
		frames = []
		for i in range(len(xPos)):
			frames.append([])
		for wl in wlframes:
			for p in wl:
				if p[0] < r:
					i = 0
				else:
					i = p[0] // r
					i = int(i)
				frames[i].append(p)

		newframes = []
		for frame in frames:
			newframe = []
			xfr = frame[0][0]
			# newframe.append([frame[0][0],len(frame)])
			for p in frame:
				# newframe.append([p[0],len(frame)])
				# np=p.pop(0)
				if np.isclose(p[2], transomTip) and p[0]<=loa/2:
					pointType = 1
				else:
					pointType = 0
				p.append(pointType)
				del p[0]
				newframe.append(p)
			newframe.insert(0,[xfr, len(frame)])
			#newframe.reverse()
			newframes.append(newframe)
		return newframes

	def generateHSMSH(self):
		lines=[self.wlinesPos,self.wlinesNeg,self.wlKeel]
		mesh=self.genHullFormMeshPP(lines)
		return mesh

	def _genFaces(self, mesh: om.PolyMesh, whs: list, doReverse: bool):
		nl = len(whs)
		npt = len(whs[0])
		for iL in range(1, nl):
			npt_iL = len(whs[iL])
			npt_iL_1 = len(whs[iL - 1])
			dip = 0
			if npt_iL > npt_iL_1:
				if doReverse:
					mesh.add_face(whs[iL][0],whs[iL - 1][0], whs[iL][1] )
					mesh.add_face(whs[iL-1][0], whs[iL - 1][1], whs[iL][1])
					mesh.add_face(whs[iL - 1][1], whs[iL][2], whs[iL][1])
				else:
					mesh.add_face(whs[iL][1], whs[iL][0], whs[iL - 1][0])
				dip = 1
			for ipL_1 in range(dip, npt_iL_1-1):
				#ip = ipL_1 + dip
				if doReverse:
					mesh.add_face(whs[iL - 1][ipL_1], whs[iL-1][ipL_1+1], whs[iL][ipL_1+1+dip],whs[iL][ipL_1+dip])
					#mesh.add_face(whs[iL - 1][ipL_1 - 1], whs[iL][ip - 1], whs[iL][ip])
				else:
					mesh.add_face(whs[iL - 1][ipL_1 - 1], whs[iL - 1][ipL_1], whs[iL][ip])
					mesh.add_face(whs[iL - 1][ipL_1 - 1], whs[iL][ip], whs[iL][ip - 1])

	def genHullFormMeshPP(self, lines: list):
		mesh = om.PolyMesh()
		om.PolyMesh()
		wlinesPos = lines[0]  # positive y waterlines
		wlinesNeg = lines[1]  # negative y waerlines
		wlKeel = lines[2]  # keel waterline (one waterline)
		#wlinesPos.reverse()
		#wlinesNeg.reverse()

		whsPos = []
		whsNeg = []
		whsi = []
		whsPos.append(whsi)
		whsNeg.append(whsi)
		for p in wlKeel:
			whsi.append(mesh.add_vertex(p))

		for wl in wlinesPos:
			whsi = []
			whsPos.append(whsi)
			for p in wl:
				whsi.append(mesh.add_vertex(p))
		for wl in wlinesNeg:
			whsi = []
			whsNeg.append(whsi)
			for p in wl:
				whsi.append(mesh.add_vertex(p))

		self._genFaces(mesh, whsPos, True)
		#self._genFaces(mesh, whsNeg, False)

		return mesh

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