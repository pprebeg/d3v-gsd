from hullform import *

import numpy as np
import time



class Hydrostatics:
    def __init__(self, hull_form:HullForm):
        self._hf = hull_form

    def getResults(self,h,seaDensity):
        tsStart = time.perf_counter() #početak mjerenja vremena

        fvs = self._hf.mesh.fv_indices().tolist()
        points = self._hf.mesh.points().tolist()
        mesh2calcWl=self.get_tria_for_calculation(fvs,points,h)
        fvs2calcWl=mesh2calcWl[0]
        points2calcWl=mesh2calcWl[1]
        xmf = self._hf.shipdata["loa_val"]/2
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


