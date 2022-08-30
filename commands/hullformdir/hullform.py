
import openmesh as om
import os
import numpy as np
import csv
import math as Math
import time
#d3v imports
from iohandlers import IOHandler
from signals import Signals
from geometry import Geometry


from typing import Set,Dict,List

# import matplotlib.pyplot as plt

class HullFormMeshQuality:
    def __init__(self):
        self._numWL = 50
        self._numPnWLhalf = 20
        self._distPolyOrder=3

    @property
    def numPointsWLhalf(self):
        return self._numPnWLhalf

    def _getDistribution(self, maxv, minv, n, pot):
        x = [0.0] * n
        for i in range(n):
            fi = float(i)
            fn1 = float(n - 1)
            x[i] = fi ** pot / fn1 ** pot * (maxv - minv) + minv
        x.reverse()
        return x

    def genWLPositions(self,hWL_top,  hWL_bottom ):
        wlPos = self._getDistribution(hWL_top, hWL_bottom, self._numWL, self._distPolyOrder)
        return wlPos
    def genWLPositionsUsingObligatory(self,obligatoryLines:list ):
        testLines = self._getDistribution(obligatoryLines[0], obligatoryLines[-1], self._numWL, self._distPolyOrder)

        nol=len(obligatoryLines)
        wlPos=[]
        i1TL=0
        for iol in range(1,nol):
            hmax=obligatoryLines[iol-1]
            hmin = obligatoryLines[iol]
            numWL=0
            for iTL in range(i1TL,self._numWL):
                if testLines[iTL] < hmax:
                    if testLines[iTL] > hmin:
                        numWL =numWL + 1
                    else:
                        i1TL=iTL
                        break
            wlPosi = self._getDistribution(hmax, hmin, numWL+2, 1)
            for wl in wlPosi:
                if len(wlPos)==0:
                    wlPos.append(wl)
                elif wl < wlPos[-1]:
                    wlPos.append(wl)
        return wlPos


class HullForm(Geometry):
    def __init__(self, fileName):
        super().__init__()
        self.filename = fileName
        self.shipdata = {}
        self.pdecks =[]
        self.dict_decks = {}
        self.pbulkheads = []
        self.hfmq = HullFormMeshQuality()
        self.shipdata,self.pdecks,self.pbulkheads,self.dict_decks = self.readShipData()
        self.h = []  # positive y waterlines
        self.wlinesNeg = []  # negative y waerlines
        self.wlKeel = []  # keel waterline (one waterline)
        self.generateMesh()

    def generateMesh(self):

        transomTip = self.shipdata["draft_val"] * self.shipdata["td_val"]
        obligatoryWL= []
        for dh in self.pdecks:
            obligatoryWL.append(dh)
        obligatoryWL.append(transomTip)
        obligatoryWL.sort(reverse=True)

        wlPos = self.hfmq.genWLPositionsUsingObligatory(obligatoryWL)
        lines = self.hullGen(self.shipdata, wlPos, self.hfmq.numPointsWLhalf)
        self.wlinesPos = lines[0]  # positive y waterlines
        self.wlinesNeg = lines[1]  # negative y waerlines
        self.wlKeel = lines[2]  # keel waterline (one waterline)
        self.mesh = self.genHullFormMeshPP(lines)
        pass

    def getResults(self,h,seaDensity):
        tsStart = time.perf_counter() #početak mjerenja vremena

        fvs = self.mesh.fv_indices().tolist()
        points = self.mesh.points().tolist()
        mesh2calcWl=self.get_tria_for_calculation(fvs,points,h)
        fvs2calcWl=mesh2calcWl[0]
        points2calcWl=mesh2calcWl[1]
        xmf = self.shipdata["loa_val"]/2
        tMesh=time.perf_counter()
        t1=tMesh-tsStart

        bcwl=self.getBasicDataUsingTrianglesProjectedToWaterline(h,xmf,fvs2calcWl,points2calcWl)
        h = bcwl[0]
        volume = bcwl[1]
        area = bcwl[2]
        Xwl = bcwl[3]
        KBz = bcwl[4]
        KBx = bcwl[5]
        Ib = bcwl[6]
        Il = bcwl[7]
        #Swet = bcwl[8]
        #print("Swet:",Swet)
        tBcwl = time.perf_counter()
        t2=tBcwl-tMesh
        Lwl, Bwl = self.getLwlBwl(h, fvs2calcWl, points2calcWl)
        mfarea = self.getMainFrameArea(xmf, h, fvs2calcWl, points2calcWl)
        hsdata = self.getHydrostaticData(seaDensity, h, volume, area, Ib, Il, KBz, Lwl, 2 * Bwl, mfarea)

        results= bcwl+hsdata
        tHSdata = time.perf_counter()
        dtAll = tHSdata - tsStart
        t3=tHSdata-tBcwl
        """print("num_triuk", len(fvs))
        print("num_tria", len(fvs2calcWl))
        print("Hydrostatic results calc timeMesh:", t1)
        print("Hydrostatic results calc timeBCWL:", t2)
        print("Hydrostatic results calc timeHSdata:", t3)

        print("Hydrostatic results calc time:", dtAll)"""
        timeuk=t1,t2,t3,dtAll
        print("time:",timeuk)
        print("Results:",results)
        print()
        return results

    def getMainFrameArea(self,x,h,fvs,points):
        mfpoints = self.getSortedPointsOnAxisAlignedPlane(x, fvs, points, 0)
        mfa=[]
        area=0
        a = len(mfpoints)
        for i in range(0,a-2,1):
                h1 = mfpoints[i][2]
                h2 = mfpoints[i+2][2]
                b1 = mfpoints[i][1]
                b2 = mfpoints[i+2][1]
                if h1<=h and h2<=h:
                    area = area + 1/2*abs(h2-h1)*(abs(b1)+abs(b2))
                    #mfa.append(1/2*abs(h2-h1)*(abs(b1)+abs(b2)))
                if h2>h and h1<h:
                    point = self.getIntersectionPoint(mfpoints[i+2],mfpoints[i],h,2)
                    H2 = point[2]
                    area = area + 1/2*abs(H2-h1)*(abs(b1)+abs(b2))
                    #mfa.append(1 / 2 * abs(h2 - h1) * (abs(b1) + abs(b2)))

        return 2*area #,np.array(mfa)

    def getTransomAngle(self, h, fvs, points):
        wlpoints = self.getSortedPointsOnAxisAlignedPlane(h, fvs, points, 2)
        wlpoints = sorted(wlpoints, key=lambda p: p[0])
        wlpoints=self.clearWlpointsList(wlpoints)
        print("wl0", wlpoints)

        deltaX = wlpoints[5][0] - wlpoints[3][0]
        deltaY=wlpoints[5][1] - wlpoints[3][1]
        angle=deltaY/deltaX
        return angle

    def clearWlpointsList(self,wlpoints):
        x=[]
        x=wlpoints
        n=len(x)
        y=[]
        #y=wlpoints
        for i in range (n):
            if x[i][1]>=0 and x[i][0]<1:
                y.append(x[i])
        return y



    def getLwlBwl(self, h, fvs, points):
        wlpoints = self.getSortedPointsOnAxisAlignedPlane(h, fvs, points, 2)
        wlpoints = sorted(wlpoints, key=lambda p: p[1])
        #print("wl1", wlpoints)
        Bwl = wlpoints[-1][1]-wlpoints[0][1]

        wlpoints = sorted(wlpoints, key=lambda p: p[0])
        Lwl = wlpoints[-1][0] - wlpoints[0][0]

        return Lwl, Bwl

    def getSortedPointsOnAxisAlignedPlane(self, x, fvs, points, os):
        mfpoints=[]
        lpr = []
        lpl = []
        for fv in fvs:
            lpr.clear()
            lpl.clear()
            for iv in fv:
                p = points[iv]
                if p[os] < x:
                    lpl.append(iv)
                elif p[os] > x:
                    lpr.append(iv)
                else:
                        mfpoints.append(p)

            if len(lpl)>0 and len(lpr) > 0:
                if len(lpl) < len(lpr):
                    mfpoints.append(self.getIntersectionPoint(points[lpl[0]],points[lpr[0]],x,os))
                    mfpoints.append(self.getIntersectionPoint(points[lpl[0]], points[lpr[1]], x, os))
                elif len(lpl) > len(lpr):
                    mfpoints.append(self.getIntersectionPoint(points[lpl[0]],points[lpr[0]],x,os))
                    mfpoints.append(self.getIntersectionPoint(points[lpl[1]], points[lpr[0]], x, os))
                else:
                    mfpoints.append(self.getIntersectionPoint(points[lpl[0]],points[lpr[0]],x,os))
                pass

        #mfpoints=[[0,1,1],[0,2,21],[1,1,2],[10,0,0]]


        import itertools
        mfpoints.sort()
        mftemp = list(mfpoints for mfpoints,_ in itertools.groupby(mfpoints))
        mfpoints = mftemp

        mfpoints = sorted(mfpoints, key=lambda p: p[2])

        return mfpoints

    def getHydrostaticData(self,seaDensity,h,volume,area,Ib, Il, KBz,Lwl,Bwl,mfarea):

        MoB =Ib / volume
        KMo = MoB + KBz
        MlB = Il / volume
        KMl = MlB + KBz
        JZ = 0.01 * area * seaDensity
        M1 = Il / Lwl
        delta = volume / seaDensity

        Cwl = area / (Lwl * Bwl)
        CB = volume / (Lwl * Bwl * h)

        CP = volume / (mfarea * Lwl)
        CX = mfarea / (Bwl * h)
        return KMo, KMl, JZ, M1, delta, Cwl, CB, CP, CX

    def getBasicDataUsingTrianglesProjectedToWaterline(self,h,xmf,fvs,points):
        fvs=np.array(fvs)
        points=np.array(points)
        k_vec = np.zeros((3))

        k_vec[2]=-1
        norm_vec, teziste = self.calc_area_cg_vector_all_tria(fvs, points)

        # area
        areaXYPlane = np.dot(norm_vec, k_vec)
        area3D = np.linalg.norm(norm_vec,axis=1)


        # Area
        Swet=area3D.sum()
        Awl =float(areaXYPlane.sum())

        # Volume
        dh =h-teziste[:,2]
        vol=areaXYPlane*dh
        Vol=float(vol.sum())

        # Xwl
        Xwl = teziste[:, 0] * areaXYPlane
        Xwl=float(Xwl.sum()/Awl)

        # Ib Il
        Ib = teziste[:,1] ** 2 * areaXYPlane
        Il = (teziste[:,0] - Xwl) ** 2 * areaXYPlane
        Ib=float(np.sum(Ib))
        Il = float(np.sum(Il))

        # Kbz, KBx
        KBz =areaXYPlane * dh * (teziste[:,2] + dh / 2)
        KBx =areaXYPlane * dh * (teziste[:,0])
        KBz = float(KBz.sum()/Vol)
        KBx=float(KBx.sum()/Vol)

        return h, 2 * Vol, 2 * Awl, Xwl, KBz, KBx, 2 * Ib, 2 * Il #,Swet

    def getIntersectionPoints(self, p1,p2,p3,h, os):
        ip1 = self.getIntersectionPoint(p1,p2,h, os)
        ip2 = self.getIntersectionPoint(p1, p3, h, os)
        ips = [ip1, ip2]
        return ips

    def getIntersectionPoint(self, p1,p2,h, os):
        ip1=0
        if os == 2:
            ip1 = [(h-p2[2])/(p1[2]-p2[2])*(p1[0]-p2[0])+p2[0], (h-p2[2])/(p1[2]-p2[2])*(p1[1]-p2[1])+p2[1] ,h]
        if os == 0:
            ip1 = [h, (h-p1[0])/(p2[0]-p1[0])*(p2[1]-p1[1])+p1[1], (h-p1[0])/(p2[0]-p1[0])*(p2[2]-p1[2])+p1[2]]

        return ip1

    def calc_area_cg_vector_all_tria(self, fvs: np.ndarray, points: np.ndarray) :
        num_tria = len(fvs)
        p1 = np.zeros((num_tria, 3))
        p2 = np.zeros((num_tria, 3))
        p3 = np.zeros((num_tria, 3))
        i = 0
        for vt in fvs:
            p1[i] = points[vt[0]]
            p2[i] = points[vt[1]]
            p3[i] = points[vt[2]]
            i += 1
        p1p2 = np.subtract(p2, p1)
        p1p3 = np.subtract(p3, p1)
        u = np.cross(p1p2, p1p3)
        cg = (p1 + p2 + p3) / 3

        return u/2,cg

    # kod za formu
    def genHullFormMesh(self, lines: list):
        mesh = om.TriMesh()
        wlinesPos = lines[0]  # positive y waterlines
        wlinesNeg = lines[1]  # negative y waerlines
        wlKeel = lines[2]  # keel waterline (one waterline)
        n1 = np.array([0,0,0])
        m1 = np.array([0,0,0])
        m2 = np.array([0,0,0])
        pt1= np.array(3)
        pt2= np.array(3)
        n2 = np.array([0,0,0])
        n3 = np.array([0,0,0])
        n4 = np.array([0,0,0])

        for i in range(len(lines) - 2):  # lijevo desno kobilica
            for j in range(len(lines[i])):  # broji vodne linije
                for k in range(len(lines[i][j]) - 2):  # broj tocaka na vodnoj liniji
                    if j == len(lines[i]) - 1:          #kobilica
                        mpt1 = lines[i][j-1][k]
                        mpt2 = lines[i][j-1][k + 1]
                        mpt3 = lines[i][len(lines[i])-1][k]
                        mpt4 = lines[i][len(lines[i])-1][k + 1]

                        # volumenA = volumenA + Voltot(mpt1,mpt2,mpt3,mpt4)
                        mesh.add_face(mesh.add_vertex(mpt1), mesh.add_vertex(mpt2), mesh.add_vertex(mpt3))
                        mesh.add_face(mesh.add_vertex(mpt2), mesh.add_vertex(mpt4), mesh.add_vertex(mpt3))

                    if j != len(lines[i])-1:         #Sve ostale vodne linije
                        mpt1 = lines[i][j][k]
                        mpt2 = lines[i][j][k + 1]
                        mpt3 = lines[i][j+1][k]
                        mpt4 = lines[i][j+1][k + 1]
                        mesh.add_face(mesh.add_vertex(mpt1), mesh.add_vertex(mpt2), mesh.add_vertex(mpt3))
                        mesh.add_face(mesh.add_vertex(mpt2), mesh.add_vertex(mpt4), mesh.add_vertex(mpt3))


        for i in range(len(lines) - 2):  # lijevo desno kobilica
                for k in range(len(lines[i][0]) - 2):  # broj tocaka na vodnoj liniji
                        pt1 = lines[i][0][k]
                        pt2 = lines[i][0][k + 1]
                        m1[0] = pt1[0]
                        m1[2] = pt1[2]
                        m2[0] = pt2[0]

                        m2[2] = pt2[2]
                        mesh.add_face(mesh.add_vertex(pt1), mesh.add_vertex(m1), mesh.add_vertex(pt2))
                        mesh.add_face(mesh.add_vertex(pt2), mesh.add_vertex(m1), mesh.add_vertex(m2))

        return mesh

    def get_tria_for_calculation(self, fvs, points, h):
        new_points = points
        new_tria = []

        lpbwl = []
        lpowl = []
        p = []
        for fh in fvs:  # facet handle
            p.clear()
            lpowl.clear()
            lpbwl.clear()
            i = 0
            lip = []
            for vh in fh:  # vertex handle
                p.append(points[vh])
                if p[i][2] > h:
                    lpowl.append(i)
                else:
                    lpbwl.append(i)
                i = i + 1

            #

            if len(lpowl) == 0:
                new_tria.append(fh)

            if len(lpowl) == 1:
                lip = self.getIntersectionPoints(p[lpowl[0]], p[lpbwl[0]], p[lpbwl[1]], h, 2)
                n = len(new_points)
                new_points.append(lip[0])
                new_points.append(lip[1])
                if lpowl[0]==1:
                    fh_new = np.array([n, n + 1, fh[lpbwl[1]]])
                    new_tria.append(fh_new)
                    fh_new = np.array([n, fh[lpbwl[1]],fh[lpbwl[0]]])
                    new_tria.append(fh_new)


                else:
                    fh_new = np.array([n, fh[lpbwl[1]], n + 1])
                    new_tria.append(fh_new)
                    fh_new = np.array([n, fh[lpbwl[0]], fh[lpbwl[1]]])
                    new_tria.append(fh_new)

            if len(lpowl) == 2:
                lip = self.getIntersectionPoints(p[lpbwl[0]], p[lpowl[0]], p[lpowl[1]], h, 2)
                n = len(new_points)
                new_points.append(lip[0])
                new_points.append(lip[1])
                if lpbwl[0] == 1:
                    fh_new = np.array([fh[lpbwl[0]], n+1, n])
                    new_tria.append(fh_new)

                else:
                    fh_new = np.array([fh[lpbwl[0]], n, n + 1])
                    new_tria.append(fh_new)

        return new_tria, new_points

    def _genFaces(self,mesh:om.TriMesh,whs:list, doReverse:bool):
        nl=len(whs)
        npt=len(whs[0])
        for iL in range(1, nl):
            npt_iL = len(whs[iL])
            npt_iL_1 = len(whs[iL-1])
            dip=0
            if npt_iL > npt_iL_1:
                if doReverse:
                    mesh.add_face(whs[iL][0], whs[iL][1], whs[iL - 1][0])
                else:
                    mesh.add_face(whs[iL][1], whs[iL][0], whs[iL - 1][0])
                dip = 1
            for ipL_1 in range(1,npt_iL_1):
                ip = ipL_1+dip
                if doReverse:
                    mesh.add_face(whs[iL - 1][ipL_1 - 1], whs[iL][ip], whs[iL - 1][ipL_1])
                    mesh.add_face(whs[iL - 1][ipL_1 - 1], whs[iL][ip - 1], whs[iL][ip])
                else:
                    mesh.add_face(whs[iL - 1][ipL_1-1],   whs[iL - 1][ipL_1],whs[iL ][ip])
                    mesh.add_face(whs[iL - 1][ipL_1 - 1], whs[iL][ip],    whs[iL][ip-1])

    def genHullFormMeshPP(self, lines: list):
        mesh = om.TriMesh()
        wlinesPos = lines[0]  # positive y waterlines
        wlinesNeg = lines[1]  # negative y waerlines
        wlKeel = lines[2]  # keel waterline (one waterline)
        wlinesPos.reverse()
        wlinesNeg.reverse()

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

        self._genFaces(mesh,whsPos,True)
        self._genFaces(mesh, whsNeg,False)

        return mesh

    def hullGen(self, shipdata: dict, pdecks: list, nump,frame_positions:List[float] = None):
        # gs is the grid size of a cell, in pixels
        # Reminder to make gridsize scaled to the screen width
        # Sets hullformdir data to slider values
        shipdata["loa_val"] = shipdata["loa_val"]
        shipdata["boa_val"] = shipdata["boa_val"]

        #
        midshipsM = shipdata["ms_val"]  # Constant m in JC equation
        bowRakeM = shipdata["bow_val"]  # Constant m in JC equation
        transomM = shipdata["tr_val"]  # Constant m in JC equation
        fwdDeckM = shipdata["deck_val"]  # Constant m in JC equation

        transomBeamMax = (shipdata["boa_val"] * shipdata["tb_val"]) / 2  # Transom half beam
        transomTip = shipdata["draft_val"] * shipdata["td_val"]
        ACU = shipdata["loa_val"] * shipdata["acu_val"]
        keelFwd = shipdata["loa_val"] * shipdata["kf_val"]
        slope = shipdata["sa_val"]

        midBeam = []  # Array with midships half beam per deck
        bowRake = []  # Array with bow rake per deck
        bowRakeS = []  # Array with bow rake per deck in superstructure
        TB = 0  # Transom half beam of a deck
        transomBeam = []  # Array with transom half beam per deck
        fwdDeckMArray = []  # Array with constants m in JC equation for deck outlines
        AE = 0  # Aft end of a deck
        aftEnd = []  # Array with aft end of each deck
        aftEndS = []  # Array with aft end of each deck in superstructure
        noseConeBaseRadius = []  # See excel tool
        ogiveRadius = []  # See excel tool
        pdecks2 = []  # Array with deck positions of hull decks
        pdecks3 = []  # Array with deck positions of superstructure decks

        for i in range(len(pdecks)):  # Assign values to variables above
            if pdecks[i] <= shipdata["draft_val"]:  # For each deck that is in the hull
                midBeam.append((Math.acosh(
                    (pdecks[i] / shipdata["draft_val"]) * (Math.cosh(midshipsM * Math.pi) - 1) + 1) / (
                                            midshipsM * Math.pi)) * (shipdata["boa_val"] / 2))
                bowRake.append((Math.acosh(
                    (pdecks[i] / shipdata["draft_val"]) * (Math.cosh(bowRakeM * Math.pi) - 1) + 1) / (
                                            bowRakeM * Math.pi)) * (shipdata["loa_val"] - keelFwd))
                if pdecks[i] > transomTip:
                    TB = ((Math.acosh(((pdecks[i] - transomTip) / (shipdata["draft_val"] - transomTip)) * (
                                Math.cosh(transomM * Math.pi) - 1) + 1) / (transomM * Math.pi)) * (transomBeamMax))

                else:
                    TB = 0

                transomBeam.append(TB)
                fwdDeckMArray.append(fwdDeckM * (pdecks[i] / (shipdata[
                    "draft_val"])) + 0.001)  # Changes constant m in JC equation to make deck outlines becomes slimmer with decreasing z position (see below)
                if (pdecks[i] >= transomTip):
                    AE = (shipdata["draft_val"] - pdecks[i]) * Math.tan(slope)

                else:
                    AE = (shipdata["draft_val"] - transomTip) * Math.tan(slope) + (transomTip - pdecks[i]) * (
                                (ACU - (shipdata["draft_val"] - transomTip) * Math.tan(slope)) / transomTip)

                aftEnd.append(AE)
                pdecks2.append(pdecks[i])

            else:  # For each deck that is in the superstructure
                aftEndS.append((pdecks[i] - shipdata["draft_val"]) * Math.tan(slope))
                bowRakeS.append(shipdata["loa_val"] - ((pdecks[i] - shipdata["draft_val"]) * Math.tan(slope)) - keelFwd)
                pdecks3.append(pdecks[i])

        for i in range(len(midBeam)):  # Assign values to variables above cont.
            noseConeBaseRadius.append(midBeam[i] - transomBeam[i])
            if noseConeBaseRadius[i] > 0:
                ogiveRadius.append(
                    (Math.pow(noseConeBaseRadius[i], 2) + Math.pow((shipdata["loa_val"] / 2) - aftEnd[i], 2)) / (
                                2 * noseConeBaseRadius[i]))

            else:
                ogiveRadius.append(0)

        deckOutlinesHull = []  # Array with hull deck outline x, y coordinates
        # Get y points for every x
        for idk in range(len(midBeam)):  # For each deck in hull
            deckOutlinesHull.append([])  # For each deck create array
            if pdecks2[idk] != 0:  # If not keel
                if transomBeam[idk] > 0:  # Add vertical hull line at transom
                    deckOutlinesHull[idk].append([aftEnd[idk], 0])
                kmin = aftEnd[idk]
                kmax = shipdata["loa_val"] / 2
                if frame_positions is not None:
                    klist= []
                    for x in frame_positions:
                        if x >= kmin and x <= kmax:
                            klist.append(x)
                else:
                    klist = np.linspace(kmin, kmax, nump)
                for xpt in klist:
                    deckOutlinesHull[idk].append([xpt, (
                                Math.sqrt(Math.pow(ogiveRadius[idk], 2) - Math.pow(xpt - shipdata["loa_val"] / 2, 2)) +
                                noseConeBaseRadius[idk] - ogiveRadius[idk] + transomBeam[idk])])

                kmin = shipdata["loa_val"] / 2
                kmax = keelFwd + bowRake[idk]
                klist = np.linspace(kmin, kmax, nump)
                for xpt in klist:
                    eqX = (xpt - shipdata["loa_val"] / 2) / (
                                keelFwd + bowRake[idk] - (shipdata["loa_val"] / 2))  # Value of x in JC equation
                    deckOutlinesHull[idk].append([xpt, (1 - ((Math.cosh(eqX * fwdDeckMArray[idk] * Math.pi) - 1) / (
                                Math.cosh(fwdDeckMArray[idk] * Math.pi) - 1))) * midBeam[idk]])


            else:  # If keel draw top
                kmin = aftEnd[idk]
                kmax = (keelFwd + bowRake[idk])
                klist = np.linspace(kmin, kmax, nump * 2)
                for xpt in klist:
                    deckOutlinesHull[idk].append([xpt, 0])  # Straight line

        deckOutlinesS = []  # Array with superstructure deck outline x, y coordinates
        tumblehome = []  # Superstructure tumblehome
        for n in range(len(aftEndS)):  # For each deck in superstructure
            deckOutlinesS.append([])  # For each deck create array
            tumblehome = (pdecks3[n] - shipdata["draft_val"]) * Math.tan(
                slope)  # Calculate tumblehome y offset to subtract below
            deckOutlinesS[n].append([aftEndS[n], 0])  # Add vertical hull line at transom

            kmin = aftEndS[n]
            kmax = shipdata["loa_val"] / 2
            klist = np.linspace(kmin, kmax, nump)
            for xpt in klist:
                deckOutlinesS[n].append([xpt, (
                            Math.sqrt(Math.pow(ogiveRadius[0], 2) - Math.pow(xpt - shipdata["loa_val"] / 2, 2)) +
                            noseConeBaseRadius[0] - ogiveRadius[0] + transomBeam[0] - tumblehome)])

            kmin = shipdata["loa_val"] / 2
            kmax = (keelFwd + bowRakeS[n])
            klist = np.linspace(kmin, kmax, nump)
            for xpt in klist:
                eqX = (xpt - shipdata["loa_val"] / 2) / (
                            keelFwd + bowRakeS[n] - (shipdata["loa_val"] / 2))  # Value of x in JC equation
                deckOutlinesS[n].append([xpt, (1 - ((Math.cosh(eqX * fwdDeckMArray[0] * Math.pi) - 1) / (
                            Math.cosh(fwdDeckMArray[0] * Math.pi) - 1))) * (midBeam[0] - tumblehome)])

        wlinesPos = []
        wlinesNeg = []
        wlKeel = []

        for ii in range(len(deckOutlinesS)):
            wlineP = list()
            wlineN = list()
            for item in deckOutlinesS[ii]:
                p = np.array([item[0], item[1], pdecks3[ii]])
                wlineP.append(p)
                p = np.array([item[0], -item[1], pdecks3[ii]])
                wlineN.append(p)
            wlinesPos.append(wlineP)
            wlinesNeg.append(wlineN)

        for ii in range(len(deckOutlinesHull)):

            if pdecks2[ii] != 0:
                wlineP = list()
                wlineN = list()
                for item in deckOutlinesHull[ii]:
                    p = np.array([item[0], item[1], pdecks2[ii]])
                    wlineP.append(p)
                    p = np.array([item[0], -item[1], pdecks2[ii]])
                    wlineN.append(p)
                wlinesPos.append(wlineP)
                wlinesNeg.append(wlineN)
            else:
                for item in deckOutlinesHull[ii]:
                    p = np.array([item[0], item[1], pdecks2[ii]])
                    wlKeel.append(p)

        return [wlinesPos, wlinesNeg, wlKeel]

    def readShipData(self):
        shipdata = {}
        pdecks = []
        dict_decks = {}
        pbulkheads = []
        with open(self.filename, newline='') as csvfile:
            f = csv.DictReader(csvfile)
            shipset = 0
            for row in f:  # there is only one row after header row!!!!!
                shipset = row

            shipdata["loa_val"] = float(shipset['LOA'])  # treba li učitavanje vrijednosti biti u petlji?
            shipdata["boa_val"] = float(shipset['BOA'])  # treba li učitavanje vrijednosti biti u petlji?
            shipdata['draft_val'] = float(shipset['D'])
            shipdata['shipname'] = shipset['Name']

            splitdata = str(shipset['HullData']).split(" ")
            shipdata["ms_val"] = float(splitdata[0])
            shipdata["bow_val"] = float(splitdata[1])
            shipdata["tr_val"] = float(splitdata[2])
            shipdata["deck_val"] = float(splitdata[3])
            shipdata["tb_val"] = float(splitdata[4])
            shipdata["td_val"] = float(splitdata[5])
            shipdata["acu_val"] = float(splitdata[6])
            shipdata["kf_val"] = float(splitdata[7])
            shipdata["sa_val"] = float(splitdata[8])

            shipdata["sp_val"] = float(splitdata[9])
            shipdata["cwl_val"] = float(splitdata[10])
            shipdata["lcb_val"] = float(splitdata[11])
            shipdata["cb_val"] = float(splitdata[12])
            shipdata["mc_val"] = float(splitdata[13])
            shipdata["bb_val"] = float(splitdata[14])
            shipdata["tran_val"] = float(splitdata[15])
            shipdata["ab_val"] = float(splitdata[16])

            shipdata["lwl_val"] = float(splitdata[17])
            shipdata["bwl_val"] = float(splitdata[18])
            shipdata["tf_val"] = float(splitdata[19])
            shipdata["ta_val"] = float(splitdata[20])

            shipdata["app1"] = float(splitdata[21])
            shipdata["area_app1"] = float(splitdata[22])
            shipdata["app2"] = float(splitdata[23])
            shipdata["area_app2"] = float(splitdata[24])
            shipdata["area_app3"] = float(splitdata[25])
            shipdata["area_app4"] = float(splitdata[26])
            shipdata["app5"] = float(splitdata[27])
            shipdata["area_app5"] = float(splitdata[28])
            shipdata["area_app6"] = float(splitdata[29])
            shipdata["area_app7"] = float(splitdata[30])
            shipdata["app8"] = float(splitdata[31])
            shipdata["area_app8"] = float(splitdata[32])
            shipdata["area_app9"] = float(splitdata[33])
            shipdata["area_app10"] = float(splitdata[34])
            shipdata["area_app11"] = float(splitdata[35])

            shipdata["cg_val"] = float(splitdata[36])
            shipdata["heading_val"] = float(splitdata[37])
            shipdata["amplitude_val"] = float(splitdata[38])
            shipdata["roll_val"] = float(splitdata[39])
            shipdata["damping_val"] = float(splitdata[40])
            shipdata["plr_val"] = float(splitdata[41])
            shipdata["gmt_val"] = float(splitdata[42])

        draft = shipdata["draft_val"]
        splitdata = str(shipset['DeckPos']).split(" ")
        nss = 0
        for dp in splitdata:
            z_deck = float(dp)
            pdecks.append(z_deck)
            if z_deck > draft:
                nss += 1
        splitdata = str(shipset['BHPos']).split(" ")
        for dp in splitdata:
            pbulkheads.append(float(dp))
        splitdata = str(shipset['Decks']).split(" ")
        i = 0
        for z_deck in pdecks:
            id_deck = nss - i
            dict_decks[id_deck] = i
            i += 1

        return shipdata, pdecks, pbulkheads, dict_decks
