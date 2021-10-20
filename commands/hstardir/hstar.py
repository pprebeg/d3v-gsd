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
	def __init__(self, fileName,num_frames,n_trans_sec,draught):
		super().__init__(0)
		self._hullform:HullForm = HullForm(fileName)
		self.shipdata=self._hullform.shipdata
		self._position = np.array([0., 0., 0.])
		self._centroid = np.array([self.LOA / 2, 0, 0])  # sredina samo za x za sada
		self._T=draught

		#self.mesh = self._hullform.mesh
		#self.mesh=self.genPolyMesh()
		#num_frames=300
		#n_trans_sec=100
		self.mesh=self._getHSMesh(num_frames,n_trans_sec)

	@property
	def deck_indexes(self):
	    return self._hullform.dict_decks


	def get_deck_z_index(self,deck_index):
		return self._hullform.pdecks[deck_index]

	def get_deck_z_code(self,deck_code):
		deck_index = self._hullform.dict_decks.get(deck_code)
		return self.get_deck_z_index(deck_index)

	def _getHSMesh(self, N_long=80, N_tocz=21):
		t=self.T
		self.mesh=self._genHSMesh(N_long,N_tocz,self.T)
		return self.mesh

	@property
	def centroid(self):
	    return self._centroid

	@property
	def T(self):				#waterline value, real draft
		return self._T
	@property
	def setDrfat(self,value):
		self._T=value

	@property
	def LOA(self):
	    return self.shipdata["loa_val"]

	@property
	def BOA(self):
		return self.shipdata["boa_val"]

	@property
	def D(self):								#geometry form draft
		return self.shipdata["draft_val"]

	@property
	def name(self):
		return self.shipdata["shipname"]

	@property
	def wlinesPos(self):
		wlinesPos=self._hullform.wlinesPos
		return wlinesPos
	@property
	def wlinesNeg(self):
		wlinesNeg=self._hullform.wlinesNeg
		return wlinesNeg

	@property
	def wlKeel(self):
		wlKeel=self._hullform.wlKeel
		return wlKeel

	@property
	def slope(self):
		return self.shipdata["sa_val"]

	@property
	def transomTip(self):
		transomTip=self.shipdata["draft_val"] * self.shipdata["td_val"]
		return transomTip

	@property
	def x0(self):
		x0=self.D*(1-self._hullform.shipdata["td_val"])*Math.tan(self.slope)
		return x0



	def _genHSMesh(self,N_long,N_tocz,T):

		step=self.LOA/N_long
		xPos=self.getXposition(0,self.LOA+step,step)
		xPos=list(filter(lambda x:x>self.x0,xPos))

		wlPos=self.getWLPositions(T)
		lines=self._hullform.hullGen(self.shipdata,wlPos,20,xPos)

		frames=self.genFramesfromWL(lines, xPos)
		mshfr=self.genFramesforMesh(frames,N_tocz)

		meshFrames=mshfr
		mesh=self.genHSHullFormMeshPP(meshFrames)

		return mesh

	def genFramesforMesh(self,frames:list,num_points=21):

		meshFrames=[]
		frCL=frames[0]
		transomFrame=frames[0]

		frames=frames[1:]

		mid=round(len(frames)/2)-1
		frame=frames[mid][1:]
		n1,n2=self.transveresePanelDistribution(num_points,frame)

		aft_fr=self.generatePointsforMeshAft(frCL,transomFrame,n2)
		meshFrames.append(aft_fr)

		for frame in frames:
			#frame=frames[i]
			xfr=frame[0][0]
			frame.pop(0)
			frame=np.array(frame)
			frame=np.transpose(frame)

			yf=frame[0]
			zf=frame[1]
			pTyp=frame[2]

			fr=self.genFramePointsforMesh(xfr,yf,zf,pTyp,num_points,n1)
			meshFrames.append(fr)
		return meshFrames

	def transveresePanelDistribution(self,N,frame:list): ###OBAVAZENO ZACRŠI!!!

		n0=N-3
		n1=1
		n2=n0-n1

		frame=np.array(frame)
		frame=np.transpose(frame)

		pTyp=frame[2]
		yf=frame[0]
		zf=frame[1]

		yd = np.diff(yf)
		zd = np.diff(zf)

		dist=np.sqrt(yd**2+zd**2)
		u=np.cumsum(dist)
		u=np.hstack([[0],u])
		utr=u*pTyp
		S_opt=u.max()/(N-1)
		f_min=[]

		for n1 in range(2,n0):
			#n1=i+1
			n2=N-n1
			t1=np.linspace(0,utr.max(),n1,endpoint=False)
			t2=np.linspace(utr.max(),u.max(),n2)
			t = np.concatenate((t1, t2))

			yn=np.interp(t,u,yf)
			zn=np.interp(t,u,zf)

			yd2 = np.diff(yn)
			zd2 = np.diff(zn)
			dist2 = np.sqrt(yd2 ** 2 + zd2 ** 2)
			u2 = np.cumsum(dist2)
			u2 = np.hstack([[0], u2])
			S=u2[1]
			f_min.append(abs((S_opt-S)/S_opt))



		min1=min(f_min)
		n1=f_min.index(min1)
		n2=N-n1
		return n1,n2




	def genFramePointsforMesh(self,xfr,yf:np.ndarray,zf:np.ndarray,pTyp:np.ndarray,num_points:int,n1:int):

		yd = np.diff(yf)
		zd = np.diff(zf)
		dist = np.sqrt(yd ** 2 + zd ** 2)
		u = np.cumsum(dist)
		u = np.hstack([[0], u])
		utr=u*pTyp
		#n1=6
		n2=num_points-n1

		if zf[0]>=(0.9*self.transomTip) and xfr>self.LOA/2:			#generiranje točaka na pramčanim rebrima
			t=np.linspace(0,u.max(),num_points)

		else:														#ostala rebra
			t1 = np.linspace(0, utr.max(), n1,endpoint=False)
			t2=np.linspace(utr.max(),u.max(),n2)
			t=np.concatenate((t1,t2))

		xn=np.full(num_points,xfr)
		yn = np.interp(t, u, yf)
		zn = np.interp(t, u, zf)-self.T
		frame=np.stack((xn,yn,zn),axis=1)
		return frame

	def generatePointsforMeshAft(self,frCL,transomFrame,num=15):
		frCL=np.array(frCL)
		trFrame=np.array(transomFrame)
		frCL.transpose()
		trFrame=np.transpose(trFrame)

		xf=trFrame[0]
		yf=trFrame[1]
		zf=trFrame[2]

		xd=np.diff(xf)
		yd = np.diff(yf)
		zd = np.diff(zf)

		dist=np.sqrt(xd**2+yd**2+zd**2)
		u=np.cumsum(dist)
		u=np.hstack([[0],u])

		t=np.linspace(0,u.max(),num)


		xn=np.interp(t, u, xf)
		yn = np.interp(t, u, yf)
		zn = np.interp(t, u, zf)-self.T

		frame=np.stack((xn,yn,zn),axis=1)

		#geometry condition

		diff=np.diff(frame,axis=0)
		diff=np.transpose(diff)
		tan=diff[2]/diff[1]
		ind=np.where(tan>=0.4)
		i1=np.min(ind)
		hpt=i1+1
		vpt=len(frame)-hpt
		i2=np.max(ind)

		f1=np.array_split(frame,[hpt])
		#f2=f1[0]
		#f3=np.transpose(f1[1])
		f2=np.flip(frame[:hpt,:],axis=0)
		f3=frame[i1+1:,:]

		x2=np.full((hpt,vpt),f3[:,0]).T
		y2 = np.linspace(f3[:, 1],0,hpt).T
		z2 = np.full((hpt,vpt),f3[:,2]).T

		#x3=np.transpose(np.full((i1,i1),f3[0]))
		#y3=np.transpose(np.linspace(f3[1],0,i1))
		#z3 = np.transpose(np.full((i1,i1),f3[2]))


		aft = np.stack((x2, y2, z2), axis=2)
		aft=np.insert(aft,0,f2,axis=0)
		a=np.transpose(aft,(1,0,2))
		#aft=np.stack((aft,f2))

		return a

	#Hstar geometry
	def getNewPointsonWL(self,xPos):

		wlinesPos=self.wlinesPos
		wlKeel=self.wlKeel
		draft = self.D
		transomTip = self.transomTip
		wlinesPos=self.clearWLframes(wlinesPos)

		frames=[]
		fmsh=[]
		transomFrame=[]
		cla = []
		clf = []
		for i in range(len(xPos)-1):
			frames.append([])
			fmsh.append([])
		z=0
		for wl in wlinesPos:
			k=0
			xfr=xPos[k]

			if wl[0][2]<=draft:
				cla.append(wl[0])
				clf.append(wl[-1])
				if wl[0][2]>=transomTip:
					transomFrame.append(wl[z].tolist())
					z=1

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

						frames[k].append(ip)
						fmsh[k].append([xfr,yp,zp])
						k+=1
						xfr=xPos[k]


		cla.reverse()
		wlKeel = cla + wlKeel + clf
		wlframeK = []
		k=0
		xfr=xPos[k]
		for ii in range(len(wlKeel) - 1):
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
			zp=wlframeK[k][2]

			if np.isclose(zp, transomTip):
				pointType = 1

			else:
				pointType = 0

			point=[wlframeK[k][1],wlframeK[k][2],pointType]
			frame.insert(0,point)
			fmsh[k].insert(0,[wlframeK[k][0],wlframeK[k][1],wlframeK[k][2]])
			fr = [xPos[k], len(frame)]
			frame.insert(0,fr)
			k+=1

		frames.insert(0,transomFrame)



		return frames,fmsh

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

	def getWLPositions(self, T):
		w1=np.linspace(0,self.transomTip,20,endpoint=False)
		w2=np.linspace(self.transomTip,T,20)
		wPos=np.concatenate((w1,w2))
		#wPos=np.flip(wPos)
		return wPos


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


	def genFramesfromWL(self, lines:list, xPos:list):

		draft=self.D
		transomTip=self.transomTip

		wlinesPos=lines[0]
		wlinesNeg=lines[1]
		wlKeel=lines[2]

		xmax=wlinesPos[-1][-1][0]
		xPos = list(filter(lambda x: x <= xmax, xPos))


		frames=[]
		transomFrame=[]
		#frCL=[]
		cla = []
		clf = []
		k=0
		for i in range(len(xPos)):
			frames.append([])

		for wl in wlinesPos:

			if wl[0][2]<=draft:
				cla.append(wl[0])
				clf.append(wl[-1])
				wl.pop(-1)
				if wl[0][2]>transomTip:
					#frCL.append(wl[0].tolist())
					transomFrame.append(wl[1].tolist())
					wl.pop(0)
					wl.pop(0)
				elif wl[0][2]==transomTip:
					transomFrame.append(wl[0].tolist())
					wl.pop(0)
				elif wl[0][2]<transomTip:
					wl.pop(0)



			for i in range(0,len(wl)):
				p=wl[i]
				xp=p[0]
				yp=p[1]
				zp=p[2]

				if np.isclose(zp,transomTip):
					pointType=1
				else:
					pointType=0
				point=[yp,zp,pointType]

				ind=xPos.index(xp)
				frames[ind].append(point)

		cla.reverse()
		wlKeel = cla + wlKeel + clf
		wlK=np.array(wlKeel)
		x=wlK[:,0]
		xK=np.array(xPos)
		yK=np.interp(xK,x,wlK[:,1])
		zK=np.interp(xK,x,wlK[:,2])
		wlK=np.stack((xK,yK,zK),axis=1)
		wlK=np.delete(wlK,-1,axis=0)


		k=0
		for p in wlK:
			yp=p[1]
			zp=p[2]
			if np.isclose(zp, transomTip):
				pointType = 1
			else:
				pointType = 0
			point = [yp, zp, pointType]
			frames[k].insert(0,point)
			k+=1

		k=0
		for frame in frames:
			xfr=xPos[k]
			num_p=len(frame)
			frame.insert(0,[xfr,num_p])
			k+=1

		bowfr=[]
		bowp=clf[-1]
		bowfr.insert(0,[bowp[0],1])
		bowfr.append([bowp[1],bowp[2],0])
		frames[-1]=bowfr

		frames.insert(0,transomFrame)

		return frames

	# Form modification
	def setShipData(self):
		pass

	def genHSHull(self):
		pass

	#PolyMesh mesh
	def genPolyMesh(self):
		lines=[self.wlinesPos,self.wlinesNeg,self.wlKeel]
		mesh=self.genPolyHullFormMeshPP(lines)
		return mesh

	def _genPolyFaces(self, mesh: om.PolyMesh, whs: list, doReverse: bool):
		nl = len(whs)
		npt = len(whs[0])
		for iL in range(1, nl):
			npt_iL = len(whs[iL])
			npt_iL_1 = len(whs[iL - 1])
			dip = 0
			if npt_iL > npt_iL_1:
				if doReverse:
					mesh.add_face(whs[iL - 1][0],whs[iL][0], whs[iL][1] )
					mesh.add_face(whs[iL-1][0], whs[iL][1], whs[iL - 1][1])
					mesh.add_face(whs[iL - 1][1], whs[iL][1], whs[iL][2])
				else:
					mesh.add_face(whs[iL - 1][0], whs[iL][1], whs[iL][0] )
					mesh.add_face(whs[iL-1][0], whs[iL - 1][1], whs[iL][1])
					mesh.add_face(whs[iL - 1][1], whs[iL][2], whs[iL][1])
				dip = 1
			for ipL_1 in range(dip, npt_iL_1-1):
				if doReverse:
					mesh.add_face(whs[iL - 1][ipL_1], whs[iL][ipL_1+dip], whs[iL][ipL_1+1+dip], whs[iL-1][ipL_1+1])
				else:
					mesh.add_face(whs[iL - 1][ipL_1], whs[iL-1][ipL_1+1], whs[iL][ipL_1+1+dip], whs[iL][ipL_1+dip])


	def genPolyHullFormMeshPP(self, lines: list):
		mesh = om.PolyMesh()
		om.PolyMesh()
		wlinesPos = self.clearWLframes(lines[0])  # positive y waterlines
		wlinesNeg = self.clearWLframes(lines[1])  # negative y waerlines
		wlKeel = lines[2]  # keel waterline (one waterline)
		rlines=wlinesPos
		rlines.insert(0,wlKeel)
		rlines=self.reverseZcoordinate(rlines)
		wlKeel = rlines[0]
		wlinesPos=rlines
		wlinesPos.pop(0)

		#wlinesPos=self.reverseZcoordinate(wlinesPos)

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
#			for p in wl:
#				whsi.append(mesh.add_vertex(p))

		self._genPolyFaces(mesh, whsPos, True)
		#self._genFaces(mesh, whsNeg, False)

		return mesh

	def reverseZcoordinate(self,wlines):
		draft=self.D
		vl=[]
		#wlines.tolist()
		for wl in wlines:
			points=[]
			for p in wl:
				point=[]
				zp=p[2]-draft
				point=[p[0],p[1],zp]
				points.append(point)
			vl.append(points)

		return vl


	#Hstra -hsmsh generatoion (faces/generator)

	def _genHSFaces(self, mesh: om.PolyMesh, whs: list,ahs:list, doReverse: bool):

		#aft mesh generation
		nl=len(ahs)
		npt=len(ahs[0])
		f0=[]
		#f0.append(ahs[0][0])
		for iL in range(1,nl):
			npt_iL=len(ahs[iL])
			#npt_iL_1 = len(ahs[iL + 1])
			f0.append(ahs[iL-1][0])
			for ipL in range(0,npt_iL-1):
				if doReverse:
					mesh.add_face(ahs[iL-1][ipL], ahs[iL-1][ipL+1],ahs[iL][ipL+1],ahs[iL][ipL])

				else:
					mesh.add_face(ahs[iL-1][ipL], ahs[iL-1][ipL+1],ahs[iL][ipL+1],ahs[iL][ipL])

		f0=f0+ahs[-1]
		f1=whs[0]
		npf0=len(f0)
		npf1=len(f1)
		ntrf1=npf1-npf0

		for i in range(0,ntrf1):
			if doReverse:
				mesh.add_face(f0[0],f1[i+1],f1[i])
			else:
				mesh.add_face(f0[0], f1[i], f1[i + 1])

		i=0
		for ii in range(ntrf1,npf1-1):
			if doReverse:
				mesh.add_face(f0[i],f0[i+1],f1[ii+1],f1[ii])
			else:
				mesh.add_face(f0[i],f0[i+1],f1[ii+1],f1[ii])
			i+=1





		#middle mesh generation
		nl = len(whs)
		npt = len(whs[0])
		for iL in range(1, nl-1):
			npt_iL = len(whs[iL])
			npt_iL_1 = len(whs[iL - 1])
			dip = 0
			for ipL_1 in range(dip, npt_iL_1-1):
				if doReverse:
					mesh.add_face(whs[iL - 1][ipL_1], whs[iL-1][ipL_1+1], whs[iL][ipL_1+1], whs[iL][ipL_1])

				else:
					mesh.add_face(whs[iL - 1][ipL_1], whs[iL-1][ipL_1+1], whs[iL][ipL_1+1], whs[iL][ipL_1])

		#bow peak mesha generation
		npt_iL_1=len(whs[-2])
		for ipL_1 in range(0, npt_iL_1 - 1):
			if doReverse:
				mesh.add_face(whs[-2][ipL_1], whs[-2][ipL_1 + 1],whs[-1][0])

			else:
				mesh.add_face(whs[-2][ipL_1], whs[-2][ipL_1 + 1],whs[-1][0])

	def genHSHullFormMeshPP(self, frames: list): #Hstar mesh generator
		mesh = om.PolyMesh()
		om.PolyMesh()
		aft=np.flip(frames[0],axis=0)
		frames=frames[1:]
		frames[-1]=[frames[-1][0]]


		ahsPos=[]
		fhsPos = []
		whsNeg = []
		whsi = []
		ahsi=[]
		#fhsPos.append(whsi)
		#ahsPos.append(ahsi)
		#whsNeg.append(whsi)

		for frame in frames:
			whsi = []
			fhsPos.append(whsi)
			for p in frame:
				whsi.append(mesh.add_vertex(p))

		for line in aft:
			ahsi=[]
			ahsPos.append(ahsi)
			for p in line:
				ahsi.append(mesh.add_vertex(p))

		"""for wl in wlinesNeg:
			whsi = []
			whsNeg.append(whsi)
		#			for p in wl:
		#				whsi.append(mesh.add_vertex(p))
		whsPos_2=wlinesPos
		whsPos_2.insert(0,wlKeel)""" #negative frames, additional option

		self._genHSFaces(mesh, fhsPos,ahsPos, True)
		# self._genFaces(mesh, whsNeg, False)

		return mesh  #



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

	@property
	def num_trans_sections(self):
	    return self._num_trsec

	@num_frames.setter
	def num_trans_sections(self, new_value):
		self._num_trsec = new_value


	def read_problem(self):
		with open(self._filename, "r") as csv_file:
			csv_reader = csv.DictReader(csv_file)
			for row in csv_reader:  # each row contains 1 block data
				hull_form_input = str(row["hull_form_file"])
				n_fr = int(row["num_frames"])
				n_trp = int(row["num_transverse_points"])
				draught = float(row["calculation_draught"])

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

			return HstarHullForm(hull_form_input,n_fr,n_trp,draught)