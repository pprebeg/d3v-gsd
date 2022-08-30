import openmesh as om
import numpy as np
import copy
import sys
import csv
import time
from functools import reduce  # Required in Python 3
import operator
#import warnings


"""
https://numpy.org/doc/stable/user/basics.broadcasting.html
When operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing dimensions and works its way forward. Two dimensions are compatible when

they are equal, or

one of them is 1

If these conditions are not met, a ValueError: operands could not be broadcast together exception is thrown, indicating that the arrays have incompatible shapes. The size of the resulting array is the size that is not 1 along each axis of the inputs.

Arrays do not need to have the same number of dimensions. For example, if you have a 256x256x3 array of RGB values, and you want to scale each color in the image by a different value, you can multiply the image by a one-dimensional array with 3 values. Lining up the sizes of the trailing axes of these arrays according to the broadcast rules, shows that they are compatible:

Image  (3d array): 256 x 256 x 3
Scale  (1d array):             3
Result (3d array): 256 x 256 x 3
When either of the dimensions compared is one, the other is used. In other words, dimensions with size 1 are stretched or “copied” to match the other.

In the following example, both the A and B arrays have axes with length one that are expanded to a larger size during the broadcast operation:

A      (4d array):  8 x 1 x 6 x 1
B      (3d array):      7 x 1 x 5
Result (4d array):  8 x 7 x 6 x 5
Here are some more examples:

A      (2d array):  5 x 4
B      (1d array):      1
Result (2d array):  5 x 4

A      (2d array):  5 x 4
B      (1d array):      4
Result (2d array):  5 x 4

A      (3d array):  15 x 3 x 5
B      (3d array):  15 x 1 x 5
Result (3d array):  15 x 3 x 5

A      (3d array):  15 x 3 x 5
B      (2d array):       3 x 5
Result (3d array):  15 x 3 x 5

A      (3d array):  15 x 3 x 5
B      (2d array):       3 x 1
Result (3d array):  15 x 3 x 5
"""
#vectorize tests functions:

class CustomError(Exception):
	pass


#http://geomalgorithms.com/a06-_intersect-2.html
#line_plane-inter with kombination broadcasting
def line_plane_inter2(ro, rd, pp):		#pp = plane points 
	v0 = pp[:,0,:]
	u = pp[:,1,:] - v0 
	v = pp[:,2,:] - v0 
	n = np.cross(u,v)
	w = vclone(v0, ro) - vadjdims(ro, v0)
	a = (n*w).sum(2)
	d = (vclone(n, rd)*vadjdims(rd, n)).sum(2)
	d = np.where(d != 0, d, np.nan)
	
	t = a / d
	tv = vadjdims(t, rd, 2)		#pos je, u ovom slucaju, sa kojim elementom mnozimo	
	delta = np.expand_dims(rd, 1)*tv
	pi = vadjdims(ro,delta,1 ) + delta  # USE .BROADCAST AND .BROADCAST_TO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
	return pi

#line_plane-inter linear (for use in tri_plane_inter2) intersects segments at 0 < t < 1
def line_plane_inter3(tp, pp):		#assumption is that tp and pp are in pairs with regards to index 0 and they have intersections
	# print("line_plane_inter was called!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	# print(tp.shape[0], pp.shape[0])
	# tpt = np.array([[[11,6.5,7.5],[0,6.5,7.5],[0,6.5,10]]])
	# ppt = np.array([[[0.16342165,0,8.75],[0,4.95,10],[0.16342165,4.7095108,8.75]]])
	# m = 162
	# g =  163
	# tp = tp[m:g]
	# pp = pp[m:g]
	# print(tp == tpt, pp == pp)
	
	# print(tp,pp)
	ro = tp
	#shift array index by 1 So ray direction can be subtrated p[1]-p[0] etc.
	a = np.expand_dims(tp[:,1],1)
	b = np.expand_dims(tp[:,2],1)
	c = np.expand_dims(tp[:,0],1)
	s = np.concatenate((a,b,c), axis = 1)
	rd = s - ro
	# print("rd")
	# print(rd)

	v0 = pp[:,0,:]
	v0v = np.expand_dims(v0, 1)
	v0v = np.concatenate((v0v,v0v,v0v), axis = 1)
	w = v0v - ro
	u = pp[:,1,:] - v0 
	v = pp[:,2,:] - v0 
	n = np.expand_dims(np.cross(u,v), 1)
	n = np.concatenate((n,n,n), axis = 1)
	dot = (n*w).sum(2)
	den = (n*rd).sum(2) 
	den = np.where(den != 0, den, np.nan)		#replace 0 with nan
	t = np.expand_dims((dot/den), 2)
	with np.errstate(invalid = "ignore"):		#removes warnings of runtime errors encountering nan-s witch works as intended
		t[t < 0.0] = np.nan		#replace bad t-s with nan-s
		t[t > 1.0] = np.nan
	pi = ro + rd*t
	#intersection point cleaning:
	sets = np.full((pi.shape[0],2,3), 0.0)
	original_indices = set(range(pi.shape[0]))
	bool = np.isnan(pi)
	nan_i = bool.all(2).any(1)	#dofs where set has nan
	dup_i = ~nan_i
	# dup_i = (~bool).all(2).all(1) 	#dofs with duplicates from vertex intersections, have no nans
	# print(pi)
	# print(bool)
	# print(nan_i, dup_i)

	# if nan_i.any():
	nan_pairs = pi[nan_i]
	i = ~np.isnan(nan_pairs).all(2)
	sets[nan_i] = nan_pairs[i].reshape(-1,2,3)
	
	if dup_i.any():
		dup_pairs = pi[dup_i]
		# print(dup_pairs)
		dup_pairs = unique_close2(dup_pairs, axis = 1).reshape(-1,2,3) # ako nije dobro; stavi unique_close(duplikati se desavaju kada je t = 0 i t=1 na 2 raya u istom verteksu trokuta; ako nece biti dobro napravi zasebni uvijet pretrazivanja u tri_tri_inter)
		# print("\n")
		# print(dup_pairs)
		# print(dup_pairs.reshape(-1,2,3).shape,dup_pairs.reshape(-1,2,3))
		# a = copy.copy(dup_pairs).reshape(-1,2,3)
		# print(a)
		# print(sets, dup_i, a)
		sets[dup_i] = dup_pairs
	
	#check if all pairs were intersected
	nanset = set(np.where(nan_i)[0])
	dupset = set(np.where(dup_i)[0])

	if len(nanset) + len(dupset) == len(original_indices):
		return sets
	else:
		error_index = original_indices - (nanset + dupset)
		raise CustomError("No intersection for pair at index: " + str(error_index))
	
	
def prod(iterable):
    return reduce(operator.mul, iterable, 1)
		
def vclone(arr,x, pos = 0):
	if pos == -1: 
		n = prod(x.shape)
	elif type(pos) == int:
		n = x.shape[pos]
	else:
		n = prod((np.asarray(x.shape)[pos]).tolist())
	return np.asarray([arr]*n)

def vadjdims(arr,x,pos = 1):
	a = len(arr.shape)
	b = len(x.shape)
	if a == b:
		n = 1
	else:
		n = b - a
	for i in range(n):
		arr = np.expand_dims(arr,pos)
	return arr
	
	
def is_in_tri2(fp, p, pair_mode = False):		#rezultat je bool array; axis 0 su pointovi, a axis 1 su trokuti, npr [2][0] je lezi li point 2 u trokutu 0, no ako je pair mode True; racunati ce trokute i pointove na istim indeksima  
	v0 = fp[:,0,:]
	u = fp[:,1,:] - v0 
	v = fp[:,2,:] - v0 
	
	if pair_mode == False:
		w = vadjdims(p, v0) - vclone(v0, p)
	elif pair_mode == True:
		if fp.shape[0] != p.shape[0]:		#if pairs do not have matching shapes
			raise CustomError("For pair mode triangle points array and points array must have same shape[0]!")
		
		w = p - v0
	
	dist = dist_p_plane2(p, fp, pair_mode)
	#print(dist)
	uu = (u**2).sum(-1)
	vv = (v**2).sum(-1)
	uv = (u*v).sum(-1)
	wu = (w*u).sum(-1)
	wv = (w*v).sum(-1)
	
	d = uv**2 - uu * vv		#denominator
	si = (uv*wv - vv*wu) / d
	ti = (uv*wu - uu*wv) / d
	siti = si+ti
	mintol = -0.00001								#povecaj tol ako treba? 
	maxtol = 1.00001
	# stavi errstate za supressanje nezeljenih warninga
	with np.errstate(invalid = "ignore"):		#removes warnings of runtime errors encountering nan-s witch works as intended
		bool = (si != np.nan) & (ti != np.nan) & (siti != np.nan) & (mintol < si) & (si < maxtol) & (mintol < ti) & (ti < maxtol) & (siti <= maxtol)
		#kada je pair mode bool ima 1 dim a dist ima 2 pa se nemoze indeksirat
		if len(dist.shape) > len(bool.shape):
			bool = np.expand_dims(bool, 0)
		bool[dist != 0] = False
		return bool
		
		# warnings.filterwarnings('ignore')
#racuna jesu li tocke blizu trokuta (dist tocke od ravnine mora biti ispod dist_tol i mora biti unutar prizme koje r)
#provijerava samo sa pozitivne strane lica
def is_in_triangular_prism2(fp, p, dist = None, dist_tol = 0.1, pair_mode = False):		#rezultat je bool array; axis 0 su pointovi, a axis 1 su trokuti, npr [2][0] je lezi li point 2 u trokutu 0, no ako je pair mode True; racunati ce trokute i pointove na istim indeksima  
	v0 = fp[:,0,:]
	u = fp[:,1,:] - v0 
	v = fp[:,2,:] - v0 
	
	#project points onto plane for further calculation:
	triangle_normals = np.cross(u,v)
	triangle_normals = triangle_normals/np.expand_dims(((triangle_normals**2).sum(-1))**0.5, 0).T
	
	if dist is None:
		dist = dist_p_plane2(p, fp, pair_mode)

	# print(triangle_normals)
	# print(dist)
	
	if pair_mode == False:
		pp = np.expand_dims(p,1) - np.expand_dims(dist,-1)*np.expand_dims(triangle_normals, 0)			#pp is projected points onto each face
		w = vadjdims(pp, v0) - vclone(v0, pp)
		
	elif pair_mode == True:
		if fp.shape[0] != p.shape[0]:		#if pairs do not have matching shapes
			raise CustomError("For pair mode triangle points array and points array must have same shape[0]!")
		pp = p - dist.T*triangle_normals
		w = pp - v0
	
	uu = (u**2).sum(-1)
	vv = (v**2).sum(-1)
	uv = (u*v).sum(-1)
	wu = (w*u).sum(-1)
	wv = (w*v).sum(-1)
	
	d = uv**2 - uu * vv		#denominator
	si = (uv*wv - vv*wu) / d
	ti = (uv*wu - uu*wv) / d
	siti = si+ti
	mintol = -0.01								#povecaj tol ako treba? 
	maxtol = 1.01
	# stavi errstate za supressanje nezeljenih warninga
	with np.errstate(invalid = "ignore"):		#removes warnings of runtime errors encountering nan-s witch works as intended
		bool = (si != np.nan) & (ti != np.nan) & (siti != np.nan) & (mintol < si) & (si < maxtol) & (mintol < ti) & (ti < maxtol) & (siti <= maxtol)	#provijera jeli projekcija po normali unutar trokuta
		# print(pp)
		# print(dist_p_plane2(pp, fp, pair_mode))
		# print(bool)
		# print(dist <= dist_tol, dist <= -1e-08)
		# dist_bool = (dist <= dist_tol) & (dist <= -1e-08)			#provijera jeli na dozvoljenoj visini prizme
		dist_bool = (np.abs(dist) <= dist_tol)
		#kada je pair mode bool ima 1 dim a dist ima 2 pa se nemoze indeksirat
		if len(dist.shape) > len(bool.shape):
			bool = np.expand_dims(bool, 0)
		# print(dist)
		# print(dist_bool)
		return_bool = bool & dist_bool
		i = np.where(return_bool)
		# print(i)
		# print(return_bool)
		# print(np.where(return_bool))
		point_indices = tuple(i[0])				
		return_fh_dict = dict(zip(point_indices, [[np.empty(0, dtype = "int64")] for _ in range(len(point_indices))]))		#dict for every fh with empty array(0,3)
		return_dist_dict = dict(zip(point_indices, [[np.empty(0)] for _ in range(len(point_indices))]))		#dict for every fh with empty array(0,3)
		
		# print(return_dict)
		# print(tuple(np.asarray(i).T))
		for pair in tuple(np.asarray(i).T):
			# print(pair)
			# print(dist[pair[0], pair[1]])
			return_fh_dict[pair[0]] = np.append(return_fh_dict[pair[0]], pair[1])
			# print(return_dict[2][1])
			return_dist_dict[pair[0]] = np.append(return_dist_dict[pair[0]], dist[pair[0], pair[1]])
		# print(return_dict)
		# print(dist[i])
		# print(return_fh_dict, return_dist_dict)
		return (return_fh_dict, return_dist_dict)
		
		
		# warnings.filterwarnings('ignore')
		
		
		
#http://geomalgorithms.com/a06-_intersect-2.html			
def tri_plane_inter2(tp, pp):  #po starom ili da samo lupim line_plane inter?
	d = dist_tri_p_plane2(tp, pp)
	return_arr = np.full((tp.shape[0], pp.shape[0], 2,3), np.nan)
	# atol = (1e-08)*1.1
	# print(d)
	Tin_bool = d < 0.0
	Tin = np.where(Tin_bool, 1, 0).sum(1)
	#Tin = Tin.reshape(Tin.shape[0], 1, Tin.shape[1])
	Ton_bool = d == 0
	Ton = np.full(Ton_bool.shape, 0)
	Ton[Ton_bool] = 1
	Ton = Ton.sum(1)
	#Ton = Ton.reshape(Ton.shape[0], 1, Ton.shape[1])
	Tout_bool = d > 0.0
	Tout = np.where(Tout_bool, 1, 0).sum(1)
	#Tout = Tout.reshape(Tout.shape[0], 1, Tout.shape[1])
	#print(d)
	count_arr = np.concatenate((Tin.reshape(Tin.shape[0], 1, Tin.shape[1]),Ton.reshape(Ton.shape[0], 1, Ton.shape[1]),Tout.reshape(Tout.shape[0], 1, Tout.shape[1])), axis = 1)
	
	# index 0 = in, 1 = on, 2 = out
	#gdje nema interectiona, return je nan!!!!  tako da ne treba pretrazivati array za te slucajeve
	#where_all_in = (count_arr == np.array([[3],[0],[0]])).all(1)
	#where_all_on = (count_arr == np.array([[0],[3],[0]])).all(1)
	#where_all_out = (count_arr == np.array([[0],[0],[3]])).all(1)
	
	#on points cond = onlen == 2 and (inlen == 1 or outlen == 1), return op_p
	onp_cond = ((count_arr == np.array([[1],[2],[0]])).all(1)) | ((count_arr == np.array([[0],[2],[1]])).all(1))   # treba li ovaj cond?
	i_onp = np.where(onp_cond)
	#inter cond = inlen == 2 and outlen == 1 or elif inlen == 1 and outlen == 2   or inlen == 1 and outlen == 1 and onlen == 1, return inter
	inter_cond = ((count_arr == np.array([[2],[0],[1]])).all(1)) | ((count_arr == np.array([[1],[0],[2]])).all(1)) | ((count_arr == np.array([[1],[1],[1]])).all(1))
	i_inter = np.where(inter_cond)	#index0 = tp, index1 = pp
	#points and planes with valid intersections:
	intersecting_tp = tp[i_inter[0]]		
	intersecting_pp = pp[i_inter[1]]

	inter = line_plane_inter3(intersecting_tp, intersecting_pp)
	#print(inter)
	#print(itp,i_inter, np.asarray(i_inter))
	#ovo vraca pune trokute
	ontp = tp[i_onp[0]]
	onpp = pp[i_onp[1]]
	
	where_onp = Ton_bool.transpose(0,2,1)[i_onp]
	onp = ontp[np.where(where_onp)].reshape(-1,2,3).astype(np.float64)	#setovi tocaka koji su na planeu
	#print(onp)	
		
	#return_inter = np.append(inter, onp,0)
	return_arr[i_onp] = onp

	return_arr[i_inter] = inter
	# print(return_arr)
	indices = np.append(i_inter, i_onp, 1)		#row 0 is triangle index and below is its pair plane index
	# print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")

	# print((return_arr[:,1] - return_arr[:,0]))
	# print(np.where(np.isclose((return_arr[:,1] - return_arr[:,0]),0)))
	
	
	return (return_arr, indices)

	

def tri_tri_inter2(tri1, tri2):
	#print(str((tri1, tri2)))
	data1 = tri_plane_inter2(tri1, tri2)
	inter1 = data1[0]
	indices1 = data1[1].T
	#print(inter1)
	
	#print(str((tri2, tri1)))
	data2 = tri_plane_inter2(tri2, tri1)
	inter2 = data2[0]
	indices2 = data2[1].T
	
	intersecting_indices = (intersect2d(indices1, np.flip(indices2, 1))).T
	s1 = inter1[tuple(intersecting_indices)]
	s2 = inter2[tuple(np.flip(intersecting_indices, 0))]		#dofs are flipped for data2 because triangle and plane points are reversed in order tri_plane_inter(tri2,tri1)
	
	#remove segment pairs where points of segment are same:
	# print(intersecting_indices)
	bad_i = (np.isclose(s1[:,1], s1[:,0]).all(-1)) | (np.isclose(s2[:,1], s2[:,0]).all(-1))
	s1 = np.delete(s1,bad_i,0)
	s2 = np.delete(s2,bad_i,0)
	intersecting_indices = np.delete(intersecting_indices, bad_i, 1)
	
	# print("------")
	#print(intersecting_indices.T)
	# print(s1[12],s2[12], intersecting_indices[:,12])
	data = segment_inter2(s1,s2)	#(points, segments witch had valid intersections)
	return (data[0], intersecting_indices[:,data[1]])
	
	

def segment_inter2(s1,s2): #segments are paralel and colinear, match eachother by 0 axis
	# print("s1")
	# print(s1[12])
	# print("s2")
	# print(s2.shape[0])
	
	line_vectors = s1[:,1] - s1[:,0]
	i = np.where(~np.isclose(line_vectors , 0))		#find dofs of where line vector is not 0
	
	# print(np.isclose(line_vectors , 0).all(-1))
	# print(line_vectors, i)				#tu nes krivo
	data = np.unique(i[0], return_index = True)		#only 1 point needed 
	
	unique_axis_0= np.expand_dims(data[0],0)
	return_index = data[1]
	unique_axis_1 = np.expand_dims(i[1][return_index],0)
	
	unique_i = np.append(unique_axis_0, unique_axis_1, 0)		#dofs of first points of set lines that are not 0
	v1 = s1[unique_i[0],:,unique_i[1]]		#v are projected points, never has 2 same points in row
	v2 = s2[unique_i[0],:,unique_i[1]]	
	# print("v1")
	# print(v1.shape)
	# print("v2")
	# print(v2.shape)
	
	v1min = np.min(v1,1)
	#v1min_bool = v1 == v1min.reshape(-1,1)
	
	v1max = np.max(v1,1)
	#v1max_bool = v1 == v1max.reshape(-1,1)
	
	v2min = np.min(v2,1)
	#v2min_bool = v2 == v2min.reshape(-1,1)
	
	v2max = np.max(v2,1)
	#v2max_bool = v2 == v2max.reshape(-1,1)
	# print(v2max.shape)
	
	#segments intersect if min or max of one segment is inbetween of, second check is if both points are in segments , third is to avoid segments intersectiong in only 1 point
	intersecting_segment_bool = (((v1min <= v2min) & (v2min <= v1max)) | ((v1min <= v2max) & (v2max <= v1max)) | ((v2min <= v1min) & (v1min <= v2max)) | ((v2min <= v1max) & (v1max <= v2max))) & ((v1min != v2max) & (v2min != v1max))
	# print(intersecting_segment_bool.shape)
	#cleaned segments:
	#print(intersecting_segment_i)
	if ~intersecting_segment_bool.all(): #if any segment index is false: make new s and v that intersect (happens when triangle planes intersect, but triangles dont)
		s1 = s1[intersecting_segment_bool]
		s2 = s2[intersecting_segment_bool]
		v1 = v1[intersecting_segment_bool]	
		v2 = v2[intersecting_segment_bool]	
		# v1min = v1min[intersecting_segment_i]
		# v1max = v1max[intersecting_segment_i]
		# v2min = v2min[intersecting_segment_i]
		# v2max = v2max[intersecting_segment_i]
		# v1min_bool = v1min_bool[intersecting_segment_i]
		# v1max_bool = v1max_bool[intersecting_segment_i]
		# v2min_bool = v2min_bool[intersecting_segment_i]
		# v2max_bool = v2max_bool[intersecting_segment_i]
		
	v = np.append(v1,v2, 1)		#projection unity points
	s = np.append(s1,s2, 1)
	# print("v")
	# print(v)
	# print("s")
	# print(s)	
	vmin = np.min(v, 1)		#arreyevi sa min i max vrijednostima
	vmax = np.max(v, 1)
	vmin_bool = ( v == vmin.reshape(-1,1))
	vmax_bool = ( v == vmax.reshape(-1,1))
	vmin_row_where_2 = np.where(vmin_bool.sum(1) == 2)[0] #where are 2 True instances
	vmax_row_where_2 = np.where(vmax_bool.sum(1) == 2)[0]


	if vmin_row_where_2.shape[0] > 0:	#if there are rows with duplicate True values 
		bad_min_i = np.argmax(vmin_bool[vmin_row_where_2],1) #argmax return first max value; in bool True is 1 ,[0] to remove from list
		vmin_bool[vmin_row_where_2, bad_min_i] = False		#changes the extra True to False
		
		
	if vmax_row_where_2.shape[0] > 0:	#if there are rows with duplicate True values 
		bad_max_i = np.argmax(vmax_bool[vmax_row_where_2],1)
		vmax_bool[vmax_row_where_2, bad_max_i] = False
	
	v_bool = vmin_bool + vmax_bool
	# print("vmin_bool")
	# print(vmin_bool)
	# print("vmax_bool")
	# print(vmax_bool)
	
	# print(v_bool)
	segments = s[~v_bool].reshape(-1,2,3) #remove ~ to return the union interval of segments, instead of difference
	# print(vmin_bool,vmax_bool)
	# print(v_bool)
	# print(segments)
	#print(vmin_bool, vmax_bool, v_bool, ~v_bool)
	return (segments,intersecting_segment_bool)
	#return(segments, unique)
		
#https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
def intersect2d(a,b):
	# print(A.dtype)
	# nrows, ncols = A.shape
	# dtype={'names':['f{}'.format(i) for i in range(ncols)],
		   # 'formats':ncols * [A.dtype]}
	
	# print(A.T.view(dtype).T)
	# C = np.intersect1d(A.T.view(dtype).T, B.T.view(dtype).T)
	# print(dtype)
	
	# C = 1
	# This last bit is optional if you're okay with "C" being a structured array...
	# C = C.view(A.dtype).reshape(-1, ncols)	
		
	# return C
	
	# print(a,b)
	# print(a.dtype.itemsize, a.dtype, a.shape[0])
	# av = a.T.view([('', a.dtype)] * a.shape[1]).T.ravel()
	# print(av)
	# print(b.dtype.itemsize, b.dtype, b.shape[0])
	# bv = b.T.view([('', b.dtype)] * b.shape[1]).T.ravel()
	# return np.intersect1d(av, bv).view(a.dtype).reshape(-1, a.shape[1])
	
	# tmp=np.prod(np.swapaxes(a[:,:,None],1,2)==b,axis=2)
	# return a[np.sum(np.cumsum(tmp,axis=0)*tmp==1,axis=1).astype(bool)]
	# -----------------
	#a is nxm and b is kxm
	#print(a.reshape(-1,1,a.shape[1]))
	# c = np.swapaxes(a[:,:,None],1,2)==b #transform a to nx1xm
	#print(c)
	# c has nxkxm dimensions due to comparison broadcast
	# each nxixj slice holds comparison matrix between a[j,:] and b[i,:]
	# Decrease dimension to nxk with product:
	# c = np.prod(c,axis=2)
	# print(c)
	#To get around duplicates://
	# Calculate cumulative sum in k-th dimension
	# c= c*np.cumsum(c,axis=0)
	tmp=np.prod(np.swapaxes(a[:,:,None],1,2)==b,axis=2)
	i = np.sum(np.cumsum(tmp,axis=0)*tmp==1,axis=1).astype(bool)
	return a[i]
	
def intersect2d_test(a,b):
	#a is nxm and b is kxm
	# print(a.shape,b.shape)
	# print(np.swapaxes(a[:,:,None],1,2))
	# print(np.expand_dims(a,1))
	# c = np.swapaxes(a[:,:,None],1,2)==b #transform a to nx1xm
	a = np.expand_dims(a,1)
	c = (a == b).all(2).astype(int)
	c = c*np.cumsum(c, 1)
	#print(c)
	#print(c.sum(1).astype(bool))

	
def unique_close2(a, axis = 0, return_indices = False, return_counts = False):		#a is nxm		#for axis >1 use only when all instances have a duplicate; else they won't be deleted to conserve array shape 
	#b = np.expand_dims(a,axis + 1) 	#b is  nx1xm for broadcasting
	#b = np.expand_dims(a,axis = (1,3))
	#b = a
	#a = a.reshape((5,1,3,3))
	#b = b.reshape(5,3,1,3)
	#print(b.shape)
	a1 = np.expand_dims(a, axis + 1)		#osigurava multidim broadcasting
	a2 = np.expand_dims(a, axis)
	isclose_bool = np.isclose(a1,a2).all(tuple(range(axis + 2, len(a1.shape))))		#bool array dim reduction .all(extra axes)
	n_of_occurance = isclose_bool.sum(axis)		#?????????
	#c = c*np.tril(c)		#remove upper tri duplicates		#if triu the last instance of duplicate array will be removed
	isclose_bool = np.tril(isclose_bool)		#remove upper tri duplicates		#if triu the last instance of duplicate array will be removed
	diag_indices = np.diag_indices(isclose_bool.shape[axis])
	
	if axis > 0:	#no need for multidim dofs #!!!!!!!!!!!jos treba prosiriti za dim > 1
		i = a.shape[0]
		x1 = np.expand_dims(np.concatenate([np.arange(i)]*isclose_bool.shape[axis]).reshape(-1,i).T.flatten(), 0)		#axis0 array dofs
		x2 = np.concatenate([np.asarray(diag_indices)]*i, 1)
		diag_indices = tuple(np.concatenate((x1,x2),0))
	isclose_bool[diag_indices] = False
	counts = isclose_bool.sum(axis)
	isclose_bool = ~(counts.astype(bool))
	unique_values = a[isclose_bool]
	if (return_indices == False) and (return_counts == False):
		return unique_values
	else:
		return_values = [unique_values]
		if return_indices == True:
			return_values.append(isclose_bool)
		if return_counts == True:
			return_values.append(n_of_occurance[isclose_bool])

		return tuple(return_values)		#
		#
	
	#c[:,np.diag_indices(c.shape[axis])] = False		([np.new_axis]*axis )
	# print(a[c])		#
	#print(c[0,np.diag_indices(c.shape[axis])])
	
	#b = np.expand_dims(a,axis + 1) 	#b is  nx1xm for broadcasting
	#c = np.isclose(b,a).all(2)	#c is nxn symetric matrix
	# c = c*np.tril(c)		#product of c and lower tri array of c removes duplicates in upepr triangle
	# c[np.diag_indices(c.shape[0])] = 0		#removes True values from diagonal
	# c = c.sum(0).astype(bool)				#indice of extra points
	# c = ~c			#unique close dofs
	# print(a[c])
	#shorter:
	# i = np.isclose(np.expand_dims(a,1), a).all(2)
	# i = i*np.tril(i)
	# i[np.diag_indices(i.shape[0])] = 0
	# counts = i.sum(0)		#how many times a point is repeated
	# i = ~(counts.astype(bool))
	# return (a[i], i, counts)
	
#http://geomalgorithms.com/a04-_planes.html#Distance-Point-to-Plane
# def dist_p_plane(p, plane_points):
	# u = plane_points[1] - plane_points[0]
	# v = plane_points[2] - plane_points[0]
	# v0 = plane_points[0]
	# n = np.cross(u,v)
	# d = -np.dot(n, v0)		
	# dn = np.sum(n**2)**0.5	
	# return (np.sum(n*p)+d) / dn	
	
	
	#REWRITE AND MERGE WITH dist_p_plane2!!!!!!!!!!!!!!!!!!!!
	#po defaultu daje kombinacije tocaka trokuta sa planevima
	#bolje napisi na pocetku da se nemora u returnu swapaxes
def dist_tri_p_plane2(tp, pp):		#tp = nt x 3 x 3, pp = np x 3 x 3
	u = pp[:,1] - pp[:,0] 
	v = pp[:,2] - pp[:,0]
	v0 = pp[:,0]
	
	#print(pp, v0)
	
	n = np.cross(u,v)
	#tpv = vclone(tp,n)
	tpv = np.expand_dims(np.swapaxes(tp,0,1),0)		#remake to shape(1 x 3 x nt x 3)
	#print(tpv2)
	#nv = vadjdims(n,tpv)
	nv = n.reshape(n.shape[0],1,1,3)
	sum = np.sum(tpv*nv, 3)
	#sum = np.sum(tpv*nv, 3)
	#print(sum,sum2)
	d = -(n*v0).sum(1)
	d = d.reshape(d.shape[0],1,1)
	dn = ((n**2).sum(1))**0.5
	dn = dn.reshape(dn.shape[0],1,1)
	
	return np.swapaxes(((sum+d) / dn), 0, -1)

def dist_p_plane2(p, pp, pair_mode = False):		#p = nt x 3, pp = np x 3 x 3
	u = pp[:,1] - pp[:,0] 
	v = pp[:,2] - pp[:,0]
	v0 = pp[:,0]
	n = np.cross(u,v)
	n = n / ((n**2).sum(-1)**0.5).reshape(-1,1)		#jedinicni vektor

	if pair_mode == False:	
		w = np.expand_dims(p, 1) - np.expand_dims(v0,0)
		nv = np.expand_dims(n,0)
	
	elif pair_mode == True:
		if p.shape[0] != pp.shape[0]:		#if pairs do not have matching shapes
			raise CustomError("For pair mode triangle points array and points array must have same shape[0]!")
		w = p - v0
		nv = np.expand_dims(n,0)
	
	return (w*nv).sum(-1)

	
def sort_by_dist2(points, sort_by_min_dist = True):			#start point is points[0], made with numpy, points are unique
	dist = (((np.expand_dims(points, 0) - np.expand_dims(points, 1))**2).sum(-1))**0.5
	dist[np.diag_indices(dist.shape[0])] = np.nan	#remove diagonal because we dont care about dist of same points
	sorted_indices = [0]
	i = 0
	dist[:,i] = np.nan
	
	if sort_by_min_dist == True:
		for x in range(dist.shape[0] - 1):  #-1 because we assume that the start is first point in points array; this is specifically for d3v block cutting needs
			dist_array = dist[i]
			i = np.nanargmin(dist_array)
			sorted_indices.append(i)
			dist[:,i] = np.nan
	
	elif sort_by_min_dist == False:
		for x in range(dist.shape[0] - 1):  #-1 because we assume that the start is first point in points array; this is specifically for d3v block cutting needs
			dist_array = dist[i]
			i = np.nanargmax(dist_array)
			sorted_indices.append(i)
			dist[:,i] = np.nan
			
	return points[np.asarray(sorted_indices)]

	
def get_neigbour_faces2(fh_idx_array, mesh,block_dims, block_position ,n = 2):
	mesh_ffi = mesh.face_face_indices()
	mesh_fvi = mesh.face_vertex_indices()
	zmin = block_position[2]
	zmax = block_position[2] + block_dims[2]
	for x in range(n):
		fh_idx_array = np.unique(np.append(fh_idx_array, mesh_ffi[fh_idx_array].flatten()))
		fh_idx_array = np.delete(fh_idx_array, np.where(fh_idx_array == -1)) #remove -1 from fvi(ffi has -1 in it to keep array shape when neighbour face number is less than max )
	#remove fh idx with points above or blow currently observed deck:
	current_vh_idx = np.unique(mesh_fvi[fh_idx_array].flatten())
	current_z = mesh.points()[current_vh_idx][:,2]
	current_fvi = mesh_fvi[fh_idx_array]
	bad_vh_idx = current_vh_idx[(current_z < zmin) | (current_z > zmax)]		#mozda promijeni tolerancije ako ne valja 
	bad_fh_idx = np.equal(np.expand_dims(current_fvi, 0), bad_vh_idx.reshape(-1,1,1)).any(0).any(-1)
	
	# print(fh_idx_array)
	# print(current_fvi)
	# print(mesh.points()[current_vh_idx])
	# print(fh_idx_array[~bad_fh_idx])
	
	return fh_idx_array[~bad_fh_idx]

	
def stitch_face2(points, original_normal): 		#[0 and following are inside points, if there are any]
	data = sort_by_angle(points, original_normal, True)
	points = np.append(np.expand_dims(data[1],0),data[0],0)		#insert centroid at 0
	n_points = points.shape[0]
	n_faces = n_points - 2
	a = (np.full(n_faces, 0)).reshape(-1,1)
	c = (np.arange(n_faces) + 1).reshape(-1,1)
	b = c + 1
	fvi = np.concatenate((a,c,b), axis = 1)
	fvi = np.append(fvi, np.expand_dims(np.array([0,(n_points-1),1]),0),0)
	# print(fvi)
	# points = np.append(np.expand_dims(centroid,0),points,0)
	# mesh_points = points[fvi]
	# normals = np.cross(mesh_points[:,1] - mesh_points[:,0], mesh_points[:,2] - mesh_points[:,0])
	# bool_normals_zero = np.isclose(normals, 0).all(-1)
	# if bool_normals_zero.any():
		# print(bool_normals_zero)
	
	
	
	return om.TriMesh(points, fvi)
	
	
	
def get_faces_near_block2(block_mesh, form_mesh, block_dims, block_position, form_mesh_avg_edge = None):
	form_mesh_vfi = form_mesh.vertex_face_indices()
	form_mesh_fvi = form_mesh.face_vertex_indices()
	form_mesh_points = form_mesh.points()
	form_mesh_tpoints = form_mesh_points[form_mesh_fvi]
	if form_mesh_avg_edge is None:
		form_mesh_edges = form_mesh_points[form_mesh.edge_vertex_indices()]
		form_mesh_edges = ((form_mesh_edges[:,1] - form_mesh_edges[:,0])**2).sum(-1)**0.5
		form_mesh_avg_edge = form_mesh_edges.sum()/form_mesh_edges.shape[0]*0.75		#inace je *1.5 ali sam podjielio sa dva jer kada radimo += da nema jos jednu operaciju
	# print(form_mesh_avg_edge)
	xmin = block_position[0] 
	xmax = block_position[0] + block_dims[0]
	ymin = block_position[1] 
	ymax = block_position[1] + block_dims[1]
	zmin = block_position[2]
	zmax = block_position[2] + block_dims[2]
	
	#get actual_inside_vh_idx
	bool = (((xmin <= form_mesh_points[:,0]) & (form_mesh_points[:,0] <= xmax)) & ((ymin <= form_mesh_points[:,1]) & (form_mesh_points[:,1] <= ymax)) & ((zmin <= form_mesh_points[:,2]) & (form_mesh_points[:,2] <= zmax)))
	form_actual_inside_vh_idx = np.where(bool)[0]		#[0] because return tuple
	
	xmin -= form_mesh_avg_edge 
	xmax += form_mesh_avg_edge
	ymin -= form_mesh_avg_edge 
	ymax += form_mesh_avg_edge
	
	bool = (((xmin <= form_mesh_points[:,0]) & (form_mesh_points[:,0] <= xmax)) & ((ymin <= form_mesh_points[:,1]) & (form_mesh_points[:,1] <= ymax)) & ((zmin <= form_mesh_points[:,2]) & (form_mesh_points[:,2] <= zmax)))
	form_quazi_inside_vh_idx = np.where(bool)[0]		#[0] because return tuple
	
	
	
	
	
	
	
	
	
	
	# xboolout = (form_mesh_tpoints > np.array([xmax, np.nan, np.nan])).any((-1,-2)) & (form_mesh_tpoints < np.array([xmin, np.nan, np.nan])).any((-1,-2))	#witch triangle fh have atleast 1 point outside of block on both sides on x axis
	# xboolin = ~xboolout
	# print(bool_xmax)			#mijenja dimenzije kak nisam predvidio
	# bool_ymax = (form_mesh_tpoints[:,:,1] > ymax).any(-1)
	# bool_ymin = (form_mesh_tpoints[:,:,1] < ymin).any(-1)
	# ybool = bool_ymax & bool_ymin
	# yboolout = (form_mesh_tpoints > np.array([np.nan, ymax, np.nan])).any((-1,-2)) & (form_mesh_tpoints < np.array([np.nan, ymin, np.nan])).any((-1,-2))	#witch triangle fh have atleast 1 point outside of block on both sides on x axis
	# yboolin = ~yboolout
	# in_fh = np.where((xboolout & yboolout) | (xboolout & yboolin) | (xboolin & yboolout))[0]
	# print(np.unique(in_fh))
	# bool_zmax = (form_mesh_tpoints[:,2] > zmax).any(-1)
	# bool_zmin = (form_mesh_tpoints[:,2] < zmin).any(-1)
	# zbool = bool_zmax & bool_zmin

	
	form_inside_fh_idx = form_mesh_vfi[form_quazi_inside_vh_idx].flatten()
	form_inside_fh_idx = np.unique(np.delete(form_inside_fh_idx, np.where(form_inside_fh_idx == -1)))	#clean -1 and remove duplicates
	
	return (form_inside_fh_idx, form_actual_inside_vh_idx)
	
def sort_by_angle(points, original_normal, return_cenroid = False):		#all points are in same plane, not colinear,unique ,first point in points is the start with angle = 0 ,orig normal is normal of ogriginal face to witch it orients the points
	# centroid = points.sum(0)/points.shape[0]
	# point_delta_vectors = points - centroid
	# cross = np.cross(point_delta_vectors[0:1], point_delta_vectors)[1:]		#cross between all delta vectors and first vector, first is with itself so we remove it
	# with np.errstate(invalid = "ignore"):		#removes warnings of runtime errors encountering nan-s witch works as intended(vector opposite of start vector)
		# cross = cross/((cross**2).sum(-1)**0.5).reshape(-1,1)
	# print(cross)
	# cross_signs = (cross*original_normal).sum(-1)
	# cross_abs = np.full(cross.shape[0], 1)			#jedinicni vektori
	# vector_abs = (point_delta_vectors**2).sum(-1)**0.5
	# denominator = (vector_abs*vector_abs[0:1])[1:]				#ne treba umnozak za za vektor 0
	# angles = cross_signs*np.arccos(cross_abs / denominator.reshape(1,-1))
	# angles[angles < 0] += 2*np.pi
	# angles[np.isnan(cross_signs).reshape(1,-1)] = np.pi				#if start and some vector are opposite their cross is 0 and because all points are unique the angle is pi
	# i = np.argsort(angles.flatten()) + 1
	# return points[np.append(np.array([0]), i)]
	
	centroid = points.sum(0)/points.shape[0]
	point_delta_vectors = points - centroid
	point_delta_vectors_abs = (point_delta_vectors**2).sum(-1)**0.5
	point_delta_vectors_dot = (point_delta_vectors[0:1] * point_delta_vectors[1:]).sum(-1)
	denominator = (point_delta_vectors_abs[0:1] * point_delta_vectors_abs[1:])				#ne treba umnozak za za vektor 0		|a|*|b|
	cross_vectors = np.cross(point_delta_vectors[0:1], point_delta_vectors[1:])		#cross between all delta vectors and first vector, first is with itself so we remove it
	cross_vectors_abs = (cross_vectors**2).sum(-1)**0.5
	cross_signs = np.sign((cross_vectors*original_normal).sum(-1))      #dot product of original normal and cross_vectors
	arg = cross_signs * cross_vectors_abs / denominator		#goes in arcsin
	angles = np.arcsin(arg)
	#print(angles*360/(2.0*np.pi))
	#arcsin function domain corrections:
	bool_where_pdv_dot_pos = point_delta_vectors_dot < 0.0
	angles[bool_where_pdv_dot_pos] = (np.pi - np.abs(angles[bool_where_pdv_dot_pos])) * np.sign(angles[bool_where_pdv_dot_pos])		#shifts angles on opposite sides of circle
	angles[arg == 1.0] = np.pi/2
	angles[arg == -1.0] = np.pi*3/2
	angles[cross_vectors_abs == 0.0] = np.pi		#cross len is 0 only for opposite vectors
	#neg angle correction:
	angles[angles < 0.0] += 2.0*np.pi
	
	#print(angles*360/(2.0*np.pi))
	i = np.argsort(angles.flatten()) + 1
	if return_cenroid == True:
		return (points[np.append(np.array([0]), i)], centroid)
	else:
		return points[np.append(np.array([0]), i)]
	
	
	#extract mesh parametriziraj!!
	#problem kada je point blocka tocno na faceu forme
	#tolerancijom se problem ne rijesava, ruse se ostali blockovi!
	#trenutni problem je taj sto mi se facevi koji bi se trebali rezati, ne rezu
	#rade se tocke na malim udaljenostima, pa ih unique_close2 brise kod stitcha i ostalih provijera za numericke greske
	#moguce rjesenje je da se tocke koje su jako blizu forme izbace van 
def cut_mesh2(block_mesh, form_mesh, block_dims, block_position, rec = False):
	# return block_mesh
	#TREBA VEZAZI n, dist tol u prism2za parametre, i izvana izracunat L, B, avg_edge_len
	# return block_mesh
	# import matplotlib.pyplot as plt
	# from mpl_toolkits.mplot3d import Axes3D
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	
	avg_block_dim = (((block_dims**2).sum(-1))**0.5)/2
	
	
	form_mesh_edges = form_mesh.points()[form_mesh.edge_vertex_indices()]
	form_mesh_edges = ((form_mesh_edges[:,1] - form_mesh_edges[:,0])**2).sum(-1)**0.5
	form_mesh_avg_edge = form_mesh_edges.sum()/form_mesh_edges.shape[0]		#inace je *1.5 ali sam podjielio sa dva jer kada radimo += da nema jos jednu operaciju


	#get form_inside_vh_idx and extract form segment:
	form_data = get_faces_near_block2(block_mesh, form_mesh, block_dims, block_position, form_mesh_avg_edge = form_mesh_avg_edge*0.75)
	if form_data[0].size == 0:
		return block_mesh
	form_fh_idx_to_check = get_neigbour_faces2(form_data[0], form_mesh,block_dims, block_position ,n = 8)
	form_inside_vh_idx = form_data[1]
	
	
	
	
	data1 = extract_mesh2(form_fh_idx_to_check, form_mesh, form_inside_vh_idx)		#duplikati u originalnoj formi!
	# print(data1[1])
	data2 = clean_mesh(data1[0], data1[1])
	form_segment_mesh = data2[0]
	form_segment_inside_vh_idx = data2[1]
	
	
	
	# print(form_segment_inside_vh_idx)
	
	#get block_inside_vh_idx:
	block_points = block_mesh.points()
	block_fvi = block_mesh.face_vertex_indices()
	block_tpoints = block_points[block_fvi]
	form_segment_fvi = form_segment_mesh.face_vertex_indices()
	form_segment_points = form_segment_mesh.points()
	# form_segment_edges = form_segment_points[form_segment_mesh.edge_vertex_indices()]
	# form_segment_edges = ((form_segment_edges[:,1] - form_segment_edges[:,0])**2).sum(-1)**0.5
	# form_segment_avg_edge = form_segment_edges.sum()/form_segment_edges.shape[0]*0.75		#inace je *1.5 ali sam podjielio sa dva jer kada radimo += da nema jos jednu operaciju


	
	
	
	form_segment_tpoints = form_segment_points[form_segment_fvi]
	block_points_form_segment_dist = dist_p_plane2(block_points, form_segment_tpoints)
	block_inside_vh_idx = np.where((block_points_form_segment_dist <= 0.0).all(-1))[0]	
	prism_data1 = is_in_triangular_prism2(form_segment_tpoints, block_points, dist_tol = avg_block_dim*0.1)		#  return (return_fh_dict, return_dist_dict)
	prism_fh_dict1 = prism_data1[0]
	prism_data2 = is_in_triangular_prism2(block_tpoints, form_segment_points, dist_tol = form_mesh_avg_edge*0.1)		#  return (return_fh_dict, return_dist_dict)
	prism_fh_dict2 = prism_data2[0]
	prism_dist_dict2 = prism_data2[1]
	
	bad_block_vh = np.asarray(tuple(prism_fh_dict1.keys()))
	bad_form_vh = tuple(prism_fh_dict2.keys())
	
	# print(block_inside_vh_idx)
	# print(np.array(prism_data[0].keys()))
	# print(block_inside_vh_idx)
	# print(block_vh_to_be_moved)
	# print(block_points_form_segment_dist)

	
	
	
	# ljk = form_segment_mesh.points()[form_segment_inside_vh_idx]
	
	# ax.scatter(ljk[:,0],ljk[:,1],ljk[:,2], c = "purple")
	
	form_segment_tpoints = form_segment_points[form_segment_fvi] 
	form_segment_normals = np.cross(form_segment_tpoints[:,1] - form_segment_tpoints[:,0], form_segment_tpoints[:,2] - form_segment_tpoints[:,0])
	form_segment_normals = form_segment_normals/((form_segment_normals**2).sum(-1)**0.5).reshape(-1,1)
	
	# ax.scatter(block_points[[0,2,4,6]][:,0],block_points[[0,2,4,6]][:,1],block_points[[0,2,4,6]][:,2], c = "purple")
	
	if ((bad_block_vh.shape[0] > 0) or (len(bad_form_vh) > 0)) and rec == False:
		# x+, x-, y+, y-
		block_quad_vh_idx = np.array([[2,3,6,7],[0,1,4,5],[3,1,5,7],[0,2,4,6]], dtype = "int64")
		block_quad_fh_idx = np.array([[[0],[1]],[[2],[3]],[[4],[5]],[[6],[7]]], dtype = "int64")
		block_quad_normals = np.array([[1.0,0.0,0.0],[-1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,-1.0,0.0]])
		block_form_segment_average_normals = dict()
		block_quads_to_move_full = set()	#block sides to move alot when it intersects form	
		bad_prism_fh = np.array([], dtype = "int64")
		
		
		for bad_vh_idx in tuple(bad_form_vh):			#check where form segment points are very close to block
			block_prism_fh = prism_fh_dict2[bad_vh_idx] 
			block_prism_dist = prism_dist_dict2[bad_vh_idx]
			i_n = np.where(np.abs(block_prism_dist) <= 0.001 )			
			bad_prism_fh = np.append(bad_prism_fh, block_prism_fh[i_n], 0)
			
		block_quads_to_move_small = np.unique(np.where(np.equal(np.unique(bad_prism_fh), block_quad_fh_idx))[0])  #checks witch block tri faces belong to witch quad id  #block sides to move alot when one of the form vertices is too close
		block_quads_to_move_small = set(list(block_quads_to_move_small))
		
		for bad_vh_idx in list(bad_block_vh):
			normals = np.delete(form_segment_normals[prism_fh_dict1[bad_vh_idx]], 2, 1).sum(0)	#removes z axis of normals and sums normal vector components
			i_n = np.argmax(np.abs(normals))			#and calcs sign of max normal vector component
			avg_max_normal = np.array([0,0])
			avg_max_normal[i_n] = np.sign(normals[i_n])
			avg_max_normal = np.append(avg_max_normal, np.array([0]) ,0)
			block_quads_to_move_full.add(np.where((block_quad_normals == avg_max_normal).all(1))[0][0])		#add witch block quad needs to be moved
		

		block_quads_to_move_small = np.asarray(tuple(block_quads_to_move_small - block_quads_to_move_full), dtype = "int64")  #ako vec strenicu pomicem puno, nema smisla da ju pomaknem malo
		block_quads_to_move_full = np.asarray(list(block_quads_to_move_full), dtype = "int64")
		block_vh_idx_to_move_full = block_quad_vh_idx[block_quads_to_move_full]
		block_vh_idx_to_move_small = block_quad_vh_idx[block_quads_to_move_small]
		
		block_normals_to_move_full = block_quad_normals[block_quads_to_move_full]
		block_normals_to_move_small = block_quad_normals[block_quads_to_move_small]
		
		if (np.abs(block_normals_to_move_full[:,0]) == 1.0).any():
			L = np.max(np.abs(np.delete(form_mesh.points(), [1,2], 1)))	#find max dims of form
		else:
			L = 0
		
		if (np.abs(block_normals_to_move_full[:,1]) == 1.0).any():
			B = np.max(np.abs(np.delete(form_mesh.points(), [0,2], 1)))
		else:
			B = 0
	
		move_vector_full = np.array([L/10,B/0.25,0]) * block_normals_to_move_full
		for i in range(block_quads_to_move_full.shape[0]):		#ovo mora preko for loopa jer corner verteksi bi se samo pomaknuli u jednom smijeru inace
			block_points[block_vh_idx_to_move_full[i]] += move_vector_full[i]
		
		move_vector_small = np.array([1e-2,1e-2,1e-2]) * block_normals_to_move_small
		for i in range(block_quads_to_move_small.shape[0]):		#ovo mora preko for loopa jer corner verteksi bi se samo pomaknuli u jednom smijeru inace
			block_points[block_vh_idx_to_move_small[i]] += move_vector_small[i]
		
		block_mesh = om.TriMesh(block_points, block_fvi)
		
		block_position =  block_points[0]
		block_dims = block_points[7] - block_points[0]
		# return block_mesh
		return cut_mesh2(block_mesh, form_mesh, block_dims, block_position, rec = True)

	
		# -------
	
		# ax.scatter(block_points[[0,2,4,6]][:,0],block_points[[0,2,4,6]][:,1],block_points[[0,2,4,6]][:,2], c = "purple")
	
		#plot uncut block
		
		# tp = block_mesh.points()[block_mesh.face_vertex_indices()]
		# v0 = tp[:,0]
		# centroids = tp.sum(1)/3
		# normals = np.cross(tp[:,1] - tp[:,0], tp[:,2] - tp[:,0])
		# tp = np.append(tp, tp[:,0].reshape(-1,1,3), 1)
		# for tri in list(tp):
			# ax.plot(tri[:,0],tri[:,1],tri[:,2], "yellow")
		# ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2], c = "red")
		
		
		
		
		
		# block_points_to_move = block_points_to_move + move_vector
		
		# for block_quad_id in block_quads_to_move
			# to_move = block_quad_vh_idx
		# print(block_quads_to_move)
		#recalc form segment inside points
		# block_tpoints = block_points[block_fvi]
		# block_points_form_segment_dist = dist_p_plane2(form_segment_points, block_tpoints)
		# print(block_points_form_segment_dist[3:4])
		# form_segment_inside_vh_idx = np.where((block_points_form_segment_dist <= 0.0).all(-1))[0]  #<= 0 jer mora biti sa unutarnje strane i tocke idu od jedne palube do druge, pa im udaljenost moze biti 0
		# print(form_segment_inside_vh_idx)
	
	
		# block_mesh = om.TriMesh(block_points, block_fvi)
		# print(block_points[block_vh_to_be_moved])
	if block_inside_vh_idx.shape[0] == 8:	#if block is completely in 
		return block_mesh
	
	
	# ljk = form_segment_mesh.points()[3:4]
	
	# ax.scatter(ljk[:,0],ljk[:,1],ljk[:,2], c = "black")
	
	
	
	#precalc block and segment normals:
	#check if block tpoints was already defined to avoid wasting time:
	
	if 'block_tpoints' not in locals():
		block_tpoints = block_points[block_fvi]
	block_normals = np.cross(block_tpoints[:,1] - block_tpoints[:,0], block_tpoints[:,2] - block_tpoints[:,0])
	block_normals = block_normals/((block_normals**2).sum(-1)**0.5).reshape(-1,1)
	
	#witch mesh faces are inside, and witch need to be cut, make data dicts with inside points 
	form_bool = np.equal(np.expand_dims(form_segment_fvi,0), form_segment_inside_vh_idx.reshape(-1,1,1))	#bool that says witch inside vh are where			
	form_bool_sum = form_bool.sum(0).sum(-1)
	form_segment_all_vh_in_fh_idx = np.where(form_bool_sum == 3)[0]
	form_segment_1_or_2_vh_inside_fh_idx = np.where((form_bool_sum == 1) | (form_bool_sum == 2))[0]
	form_ind = np.where(form_bool)				#[0] are dofs of block_inside_vh_idx, [1] are fh
	
	form_segment_data_dict = dict(zip(tuple(range(form_segment_fvi.shape[0])), [np.empty((0,3))]*form_segment_fvi.shape[0]))		#dict for every fh with empty array(0,3)
	for pair in list(np.asarray(form_ind)[0:2].T):
		form_segment_data_dict[pair[1]] = np.append(form_segment_data_dict[pair[1]], np.expand_dims(form_segment_points[form_segment_inside_vh_idx[pair[0]]], 0), 0)
	
	# form_bool2 = form_bool.any(0).all(-1)		#bool that says witch faces have all inside points
	# print(form_bool2)
	# form_segment_all_vh_in_fh_idx = np.where(form_bool2)[0]			#fh with all vertices on the inside; merge with cut block mesh later
	# form_segment_fh_idx_to_cut = np.where(~form_bool2)[0]
	
	block_bool = np.equal(np.expand_dims(block_fvi,0), block_inside_vh_idx.reshape(-1,1,1))			#bool of block seg fvi ; True means that a vi is inside
	block_ind = np.where(block_bool)			#[0] are dofs of block_inside_vh_idx, [1] are fh
	
	block_data_dict = dict(zip(tuple(range(block_fvi.shape[0])), [np.empty((0,3))]*block_fvi.shape[0]))		#dict for every fh with empty array(0,3)
	for pair in list(np.asarray(block_ind)[0:2].T):
		block_data_dict[pair[1]] = np.append(block_data_dict[pair[1]], np.expand_dims(block_points[block_inside_vh_idx[pair[0]]], 0), 0)
	
	block_bool2 = block_bool.any(0).all(-1)
	block_inside_fh_idx = np.where(block_bool2)[0]
	block_fh_idx_to_cut = np.where(~block_bool2)[0]
	
	#data dicts currently have all faces with atleast 1 inside point in them
	
	
	
	
	#cutting designated faces:
	data = tri_tri_inter2(block_points[block_fvi], form_segment_points[form_segment_fvi])
	intersection_points = data[0]
	intersection_pair_indices = data[1].T
	block_fh_to_stitch = np.unique(intersection_pair_indices[:,0])
	a = intersection_pair_indices[:,1]
	data3 = np.unique(a, return_counts = True)
	# print(a)
	form_segment_fh_to_stitch = np.unique(np.append(data3[0][data3[1] > 1],form_segment_1_or_2_vh_inside_fh_idx))		#were cut by more than 1 block face
	form_segment_fh_to_extract = np.setdiff1d(form_segment_all_vh_in_fh_idx,form_segment_fh_to_stitch ,True)
	# print(form_segment_all_vh_in_fh_idx, form_segment_fh_to_stitch, form_segment_fh_to_extract)
	# form_segment_fh_to_stitch = np.intersect1d(np.unique(intersection_pair_indices[:,1]), form_segment_1_or_2_vh_inside_fh_idx)
	# print(form_segment_fh_to_stitch)
	# print(form_segment_inside_fh_idx, form_segment_fh_to_stitch)
	
	
	
	#fill dicts with points:
	for i in range(intersection_points.shape[0]):		#pair[0] block fh idx, [1] is form fh idx
		block_data_dict[intersection_pair_indices[i][0]] = np.append(block_data_dict[intersection_pair_indices[i][0]], intersection_points[i], 0)
		form_segment_data_dict[intersection_pair_indices[i][1]] = np.append(form_segment_data_dict[intersection_pair_indices[i][1]], intersection_points[i], 0)
	
	#stitch block faces:
	if block_inside_fh_idx.shape[0] > 0:		#if there are inside faces, extract and append them to mesh list that will be merged later
		block_merge_mesh_list = [extract_mesh2(block_inside_fh_idx, block_mesh)]
	else:
		block_merge_mesh_list = []
	for block_fh in list(block_fh_to_stitch):
		points = unique_close2(block_data_dict[block_fh]) # merganje
		#points = block_data_dict[block_fh] #pp bez merganja

		# print(points.shape)
		# ax.scatter(points[:,0],points[:,1],points[:,2], c = "red")
	
		# points =block_data_dict[block_fh]
		if points.shape[0] >= 3:
			original_face_normal = block_normals[block_fh]
			# points = sort_by_angle(points, original_face_normal)
			# print("points:")
			# print(points)
			cut_mesh = stitch_face2(points, original_face_normal)
			cut_mesh_normal = cut_mesh.calc_face_normal(cut_mesh.face_handle(0))   #ovo mozda nedaje dobre normale; promijeni u np cross ako se desi
			# print("cut_mesh normal:")
			# print(cut_mesh_normal)
			# print("original face normal:")
			# print(original_face_normal, block_fh)
			if ~(np.isclose(cut_mesh_normal, original_face_normal).all()):
				cut_mesh = flip_mesh_face_orientation(cut_mesh)
				# print("flipped")
			block_merge_mesh_list.append(cut_mesh)
		
	
	#stitch form segment faces:
	if form_segment_all_vh_in_fh_idx.shape[0] > 0:
		form_segment_merge_mesh_list = [extract_mesh2(form_segment_fh_to_extract, form_segment_mesh)]		#if there are faces with all 3 points inside; extract them and put them into list to merge later
		# form_segment_merge_mesh_list = []
	else:
		form_segment_merge_mesh_list = []
	
	# COMPLEX FACES IN SEGMENT MESH; CLEAN AT BEGGINING NEW HARD MERGE
	# print(form_segment_fh_to_stitch)
	for form_fh in list(form_segment_fh_to_stitch):
		points = unique_close2(form_segment_data_dict[form_fh])
		# print(points.shape)
		# ax.scatter(points[:,0],points[:,1],points[:,2], c = "yellow")
	
		# points = form_segment_data_dict[form_fh]
		# if (points.shape[0] >= 3) and are_all_points_colinear(points) == False:		#must be atleast 3 points for face and points must not be colinear!
		if (points.shape[0] >= 3):
			original_face_normal = form_segment_normals[form_fh]
			cut_mesh = stitch_face2(points, original_face_normal)
			cut_mesh_face_points = cut_mesh.points()[cut_mesh.face_vertex_indices()[0]]
			cut_mesh_normal = np.cross(cut_mesh_face_points[1] - cut_mesh_face_points[0], cut_mesh_face_points[2] - cut_mesh_face_points[0])
			cut_mesh_normal = cut_mesh_normal/((cut_mesh_normal**2).sum(-1)**0.5).reshape(-1,1)
			
			if ~(np.isclose(cut_mesh_normal, original_face_normal).all()):
				cut_mesh = flip_mesh_face_orientation(cut_mesh)
			form_segment_merge_mesh_list.append(cut_mesh)
		
	final_mesh = hard_merge_meshes2(block_merge_mesh_list + form_segment_merge_mesh_list)
	#za plotanje:
	# mesh1 = hard_merge_meshes2( form_segment_merge_mesh_list)
	# mesh2 = hard_merge_meshes2(block_merge_mesh_list)
	
	#make new clean mesh function(with option to sync vh and fh), new stitch
	#either stitch face dosen't work as intended or form mesh has duplicates! both
	# print("block n points:")
	# print(hard_merge_meshes2(block_merge_mesh_list).points().shape[0])

	# print("form n points:")
	# print(hard_merge_meshes2(form_segment_merge_mesh_list).points().shape[0])	
	
	
	#plot intersection points
	# print(intersection_points)
	# ax.scatter(intersection_points[:,:,0],intersection_points[:,:,1],intersection_points[:,:,2], c = "black")
	
	# plot form segment
	
	# tp = form_segment_mesh.points()[form_segment_mesh.face_vertex_indices()]
	# v0 = tp[:,0]
	# centroids = tp.sum(1)/3
	# normals = np.cross(tp[:,1] - tp[:,0], tp[:,2] - tp[:,0])
	# tp = np.append(tp, tp[:,0].reshape(-1,1,3), 1)
	# for tri in list(tp):
		# ax.plot(tri[:,0],tri[:,1],tri[:,2], "blue")
	# ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2], c = "blue")
	
	#plot block inside points
	
	# tp = block_mesh.points()
	# ax.scatter(tp[:,0],tp[:,1],tp[:,2], c = "black")
	
	#plot uncut block
	
	# tp = block_mesh.points()[block_mesh.face_vertex_indices()]
	# v0 = tp[:,0]
	# centroids = tp.sum(1)/3
	# normals = np.cross(tp[:,1] - tp[:,0], tp[:,2] - tp[:,0])
	# tp = np.append(tp, tp[:,0].reshape(-1,1,3), 1)
	# for tri in list(tp):
		# ax.plot(tri[:,0],tri[:,1],tri[:,2], "yellow")
	# ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2], c = "red")
	
	# plot cut blok with boundary edges 
	# points = mesh1.points()
	# evi = mesh1.edge_vertex_indices()
	# fvi = mesh1.face_vertex_indices()
	# print(points.shape)
	# c = 0
	# for eh in mesh1.edges():
		# ep = points[evi[eh.idx()]]
		# if mesh1.is_boundary(eh):
			# c += 1
			# ax.plot(ep[:,0],ep[:,1],ep[:,2], "red")
		# else:
			# ax.plot(ep[:,0],ep[:,1],ep[:,2], "blue")
	
	# centroids = points[fvi].sum(1)/3
	# ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2], c = "orange")
	
	# ax.scatter(points[:,0],points[:,1],points[:,2], c = "purple")
	
	# points = mesh2.points()
	# evi = mesh2.edge_vertex_indices()
	# fvi = mesh2.face_vertex_indices()
	# print(points.shape)
	# c = 0
	# for eh in mesh2.edges():
		# ep = points[evi[eh.idx()]]
		# if mesh2.is_boundary(eh):
			# c += 1
			# ax.plot(ep[:,0],ep[:,1],ep[:,2], "red")
		# else:
			# ax.plot(ep[:,0],ep[:,1],ep[:,2], "green")
	
	# centroids = points[fvi].sum(1)/3
	# ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2], c = "black")
	# ax.scatter(points[:,0],points[:,1],points[:,2], c = "orange")
	
	
	
	
	# print(c)
		# normals = normals/((normals**2).sum(-1)**0.5).reshape(-1,1)
		# normals = np.append(centroids.reshape(-1,1,3), (normals + centroids).reshape(-1,1,3), 1)
	
	
	# for mesh in form_segment_merge_mesh_list:
		# tp = mesh.points()[mesh.face_vertex_indices()]
		# v0 = tp[:,0]
		# centroids = tp.sum(1)/3
		# normals = np.cross(tp[:,1] - tp[:,0], tp[:,2] - tp[:,0])
		# tp = np.append(tp, tp[:,0].reshape(-1,1,3), 1)

		# for tri in list(tp):
			# ax.plot(tri[:,0],tri[:,1],tri[:,2], "yellow")
		# ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2], c = "yellow")

	# for mesh in block_merge_mesh_list:
		# tp = mesh.points()[mesh.face_vertex_indices()]
		# v0 = tp[:,0]
		# centroids = tp.sum(1)/3
		# normals = np.cross(tp[:,1] - tp[:,0], tp[:,2] - tp[:,0])
		# tp = np.append(tp, tp[:,0].reshape(-1,1,3), 1)

		# for tri in list(tp):
			# ax.plot(tri[:,0],tri[:,1],tri[:,2], "green")
		# ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2], c = "green")





		# for n in list(normals):
			# ax.plot(n[:,0],n[:,1],n[:,2], "blue")
		# for int in list(intersection_points):
			# ax.scatter(int[:,0],int[:,1],int[:,2], c = "purple")
	# plot cut form
	# for mesh in form_segment_merge_mesh_list:
		# tp = mesh.points()[mesh.face_vertex_indices()]
		# print("aaaaa")
		# print(are_all_points_colinear(mesh.points()))
		# v0 = tp[:,0]
		# centroids = tp.sum(1)/3
		# normals = np.cross(tp[:,1] - tp[:,0], tp[:,2] - tp[:,0])
		# normals = normals/((normals**2).sum(-1)**0.5).reshape(-1,1)
		# normals = np.append(centroids.reshape(-1,1,3), (normals + centroids).reshape(-1,1,3), 1)
		# tp = np.append(tp, tp[:,0].reshape(-1,1,3), 1)
		# for tri in list(tp):
			# ax.plot(tri[:,0],tri[:,1],tri[:,2],"blue")
		# ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2], c = "blue")
		# for n in list(normals):
			# ax.plot(n[:,0],n[:,1],n[:,2], "blue")
	
		# for int in list(intersection_points):
			# ax.scatter(int[:,0],int[:,1],int[:,2], c = "purple")
	# plt.show()
	
	
	# plot_block:
	# tp = block_mesh.points()[block_mesh.face_vertex_indices()]
	# print(tp)
	# v0 = tp[:,0]
	# centroids = tp.sum(1)/3
	# normals = np.cross(tp[:,1] - tp[:,0], tp[:,2] - tp[:,0])
	# normals = normals/((normals**2).sum(-1)**0.5).reshape(-1,1)
	# normals = np.append(centroids.reshape(-1,1,3), (normals + centroids).reshape(-1,1,3), 1)
	# tp = np.append(tp, tp[:,0].reshape(-1,1,3), 1)
	# for tri in list(tp):
		# ax.plot(tri[:,0],tri[:,1],tri[:,2], "red")
		
	# ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2], c = "red")
	# for n in list(normals):
		# ax.plot(n[:,0],n[:,1],n[:,2], "red")
	# for int in list(intersection_points):
		# ax.scatter(int[:,0],int[:,1],int[:,2], c = "green")

	#plot_form:
	# tp = form_segment_mesh.points()[form_segment_fvi]
	# print(tp)
	# top_idx = np.where((tp[:,:,2] == 10).sum(-1) == 2)
	# bottom_idx = np.where((tp[:,:,2] == 10).sum(-1) == 1)
	# print(top_idx, bottom_idx)
	# v0 = tp[:,0]
	# centroids = tp.sum(1)/3
	# normals = np.cross(tp[:,1] - tp[:,0], tp[:,2] - tp[:,0])
	# normals = normals/((normals**2).sum(-1)**0.5).reshape(-1,1)
	# normals = np.append(centroids.reshape(-1,1,3), (normals + centroids).reshape(-1,1,3), 1)
	# tp = np.append(tp, tp[:,0].reshape(-1,1,3), 1)
	# for tri in list(tp):
		# ax.plot(tri[:,0],tri[:,1],tri[:,2], "green")
		
	# ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2], c = "red")
	# for n in list(normals):
		# ax.plot(n[:,0],n[:,1],n[:,2], "red")
	# for int in list(intersection_points):
		# ax.scatter(int[:,0],int[:,1],int[:,2], c = "green")

	#print(block_mesh.points(), form_segment_mesh.points())
	# bp = block_mesh.points()[block_mesh.face_vertex_indices()]
	
	
	# data = tri_tri_inter2(bp,tp)	

	# for p in list(data[0]):
		# ax.scatter(p[:,0],p[:,1],p[:,2], "blue")

	# tp = np.append(tp, tp[:,0].reshape(-1,1,3), 1)
	# for tri in list(tp):
		# ax.plot(tri[:,0],tri[:,1],tri[:,2], "green")
				
	#final merged mesh plot:
	# points = final_mesh.points()
	# evi = final_mesh.edge_vertex_indices()
	# for eh in final_mesh.edges():
		# if final_mesh.is_boundary(eh):
			# ep = points[evi[eh.idx()]]
			# ax.plot(ep[:,0],ep[:,1],ep[:,2], "red")
			
			
	# plt.show()
	#print(is_mesh_closed(final_mesh))
	return final_mesh

	
def extract_mesh2(fh_idx_to_extract, mesh, mesh_vh_idx_to_sync = None):		#mesh vh to sync are vh idx for witch we need their new idx 
	extracted_fvi = mesh.face_vertex_indices()[fh_idx_to_extract]
	extracted_vi = np.unique(extracted_fvi.flatten())
	new_points = mesh.points()[extracted_vi]
	new_fvi = replace(extracted_fvi, extracted_vi, np.arange(extracted_vi.shape[0]))
	if mesh_vh_idx_to_sync is None:
		return om.TriMesh(new_points, new_fvi)
	else:
		synced_vh = np.intersect1d(extracted_vi, mesh_vh_idx_to_sync, return_indices = True)[1] #intersect1d returns 1, common elements, 2,dofs of first occurace in arr1 3, first comm in arr2 ; new vh idx is equal to dofs of occurance in first array
		return (om.TriMesh(new_points, new_fvi), synced_vh)
	
def replace(array, values_to_replace, values_to_replace_with):			#replaces values of array with values to replace arg2 and 1 are 1D array with same shape	
	ind = np.asarray(np.where(np.equal(np.expand_dims(array,0), values_to_replace.reshape(values_to_replace.shape[0],1,1))))
	array[tuple(ind[1:])] = values_to_replace_with[ind[0]]
	return array
	
	
def flip_mesh_face_orientation(mesh):
	flipped_fvi = np.flip(mesh.face_vertex_indices(), axis = 1)
	return om.TriMesh(mesh.points(), flipped_fvi)
	

def clean_mesh(mesh, vh_idx_to_sync = None):
	if mesh.n_faces() == 0:
		return mesh
	mesh_fvi = mesh.face_vertex_indices()
	mesh_points = mesh.points()
	mesh_tpoints = mesh_points[mesh_fvi]
	mesh_normals = np.cross(mesh_tpoints[:,1] - mesh_tpoints[:,0], mesh_tpoints[:,2] - mesh_tpoints[:,0])
	bool = np.isclose(mesh_normals, 0.0).all(-1)
	mesh_fvi = np.delete(mesh_fvi, bool, 0)
	return delete_isolated_vertices2(om.TriMesh(mesh_points, mesh_fvi), vh_idx_to_sync)
	

	
		#!!! complex edge if 2 or more faces have all 3 points same
def soft_merge_meshes(meshes, vh_idx_to_sync_list = None):	#meshes je lista sa meshevima, vh_idx_to_sync_list sa lista isog lena ko meshes, svaka sadržava array sa vh_idx koji zelimo syncat
	points = np.empty((0,3))
	merged_fvi = np.empty((0,3))

	if vh_idx_to_sync_list is None:
		for mesh in meshes:
			mesh_fvi = mesh.face_vertex_indices()
			if mesh_fvi.size==0:
				continue
			merged_fvi = np.append(merged_fvi, mesh_fvi + points.shape[0], axis = 0)				#+points.shape[0] je tu da poreda face_vertex_indices sa njihovim indexom u novom arrayu
			points = np.append(points, mesh.points(), axis = 0)
		
		return clean_mesh(om.TriMesh(points, merged_fvi))	
	
	else:
		synced_vh_idx = np.empty((0), dtype = "int64")
		for i in range(len(meshes)):
			mesh = meshes[i]
			mesh_fvi = mesh.face_vertex_indices()
			merged_fvi = np.append(merged_fvi, mesh_fvi + points.shape[0], axis = 0)				#+points.shape[0] je tu da poreda face_vertex_indices sa njihovim indexom u novom arrayu
			synced_vh_idx = np.append(synced_vh_idx, (vh_idx_to_sync_list[i] + points.shape[0]), 0)
			points = np.append(points, mesh.points(), axis = 0)
		return clean_mesh(om.TriMesh(points, merged_fvi), synced_vh_idx)	
	

def hard_merge_meshes2(meshes, vh_idx_to_sync = None):			#vh_idx_to_sync_list is list with numpy arrays
	if vh_idx_to_sync is None:
		merged_mesh = soft_merge_meshes(meshes)
		merged_mesh = soft_merge_meshes(meshes)

		if merged_mesh.n_faces() ==0:
			return merged_mesh
		merged_mesh_fvi = merged_mesh.face_vertex_indices()
		merged_mesh_points = merged_mesh.points()
		
		bool = np.isclose(np.expand_dims(merged_mesh_points, 0), np.expand_dims(merged_mesh_points, 1)).all(-1)
		#clean diag and lower triangle matrix
		bool[np.diag_indices(bool.shape[0])] = False
		bool = np.triu(bool)
		ind = np.asarray(np.where(bool))
		#remove duplicates incase 3+ vh idx on same point
		data = np.unique(ind[1], return_index = True) #[0] unique values, [1] their dofs in orig array,
		ind = ind[:, data[1]]
		#delete duplicate points, replace duplicate vh_idx in fvi
		#duplicate vh_idx reduction:
		fvi_ind = np.where(np.expand_dims(merged_mesh_fvi, 0) == ind[1].reshape(-1,1,1))
		merged_mesh_fvi[fvi_ind[1:3]] = ind[0][fvi_ind[0]]		#slice fvi ind because [0] is indice of what vh_idx the fvi were compared to, 1,2 are actual dofs of fvi to be replaced
		#syncing fvi afrer deleting duplicate points:
		vh_to_delete = np.unique(ind[1])
		vh_to_keep = np.delete(np.arange(merged_mesh_points.shape[0]), vh_to_delete, 0)
		merged_mesh_points = np.delete(merged_mesh_points, vh_to_delete, 0)
		fvi_ind = np.where(np.expand_dims(merged_mesh_fvi, 0) == vh_to_keep.reshape(-1,1,1))
		merged_mesh_fvi[fvi_ind[1:3]] = fvi_ind[0]		#slice fvi ind because [0] is indice of what vh_idx the fvi were compared to, 1,2 are actual dofs of fvi to be replaced
		
		return om.TriMesh(merged_mesh_points, merged_mesh_fvi)
		
	else:
		data = soft_merge_meshes(meshes, vh_idx_to_sync)
		merged_mesh = data[0]
		vh_idx_to_sync = data[1]
		merged_mesh_fvi = merged_mesh.face_vertex_indices()
		merged_mesh_points = merged_mesh.points()
		
		bool = np.isclose(np.expand_dims(merged_mesh_points, 0), np.expand_dims(merged_mesh_points, 1)).all(-1)
		#clean diag and lower triangle matrix
		bool[np.diag_indices(bool.shape[0])] = False
		bool = np.triu(bool)
		ind = np.asarray(np.where(bool))
		#remove duplicates incase 3+ vh idx on same point
		data = np.unique(ind[1], return_index = True) #[0] unique values, [1] their dofs in orig array,
		ind = ind[:, data[1]]
		#delete duplicate points, replace duplicate vh_idx in fvi
		#duplicate vh_idx reduction:
		fvi_ind = np.where(np.expand_dims(merged_mesh_fvi, 0) == ind[1].reshape(-1,1,1))
		merged_mesh_fvi[fvi_ind[1:3]] = ind[0][fvi_ind[0]]		#slice fvi ind because [0] is indice of what vh_idx the fvi were compared to, 1,2 are actual dofs of fvi to be replaced
		#syncing fvi afrer deleting duplicate points:
		vh_to_delete = np.unique(ind[1])
		vh_to_keep = np.delete(np.arange(merged_mesh_points.shape[0]), vh_to_delete, 0)
		merged_mesh_points = np.delete(merged_mesh_points, vh_to_delete, 0)
		fvi_ind = np.where(np.expand_dims(merged_mesh_fvi, 0) == vh_to_keep.reshape(-1,1,1))
		merged_mesh_fvi[fvi_ind[1:3]] = fvi_ind[0]		#slice fvi ind because [0] is indice of what vh_idx the fvi were compared to, 1,2 are actual dofs of fvi to be replaced
		
		#sync vh idx:
		data = np.intersect1d(vh_idx_to_sync, ind[1], return_indices = True)
		vh_idx_to_sync[data[1]] = ind[0][data[2]]
		
		return (om.TriMesh(merged_mesh_points, merged_mesh_fvi), np.unique(vh_idx_to_sync))
	

def delete_isolated_vertices2(mesh, vh_idx_to_sync = None):	
	if vh_idx_to_sync is None:
		mesh_points = mesh.points()
		mesh_vfi = mesh.vertex_face_indices()
		mesh_fvi = mesh.face_vertex_indices()
		bool = (mesh_vfi == -1).all(-1)
		# delete isolated vertices:
		mesh_points = np.delete(mesh_points, bool, 0)
		#sync fvi:
		old_ind = np.delete(np.arange(bool.shape[0]), bool, 0)
		ind = np.where(np.expand_dims(mesh_fvi, 0) == old_ind.reshape(-1,1,1))
		mesh_fvi[ind[1:3]] = ind[0]
		return om.TriMesh(mesh_points, mesh_fvi)
	
	else:
		mesh_points = mesh.points()
		mesh_vfi = mesh.vertex_face_indices()
		mesh_fvi = mesh.face_vertex_indices()
		bool = (mesh_vfi == -1).all(-1)
		# delete isolated vertices:
		vi = np.arange(bool.shape[0])
		mesh_points = np.delete(mesh_points, bool, 0)
		#sync fvi:
		old_ind = np.delete(vi, bool, 0)
		ind = np.where(np.expand_dims(mesh_fvi, 0) == old_ind.reshape(-1,1,1))
		mesh_fvi[ind[1:3]] = ind[0]
		#sync vh_idx_list:
		#remove isolated vertices from sync array:
		deleted_vh_idx = vi[bool]
		vh_idx_to_sync = np.delete(vh_idx_to_sync, np.equal(np.expand_dims(deleted_vh_idx,0),np.expand_dims(vh_idx_to_sync,1)).any(-1), 0)
		data = np.intersect1d(old_ind, vh_idx_to_sync, return_indices = True)
		vh_idx_to_sync[data[2]] = data[1]				#*******************************mozda krivo, ako kasnije daje pogreske pogledaj ovdje!
		
		return (om.TriMesh(mesh_points, mesh_fvi), vh_idx_to_sync)

		
	
	
def make_block(block_dims = np.array([20,6,3]), move_vector = np.array([0,0,0])):
	# block_dims = block_dims - np.array([0,0,0.0000001])
	mesh = om.TriMesh()
	axes = []
	#stvara 2 tocke na svakoj osi
	for dim in block_dims:
		axes.append(np.linspace(0, dim, 2))
		
	block_corners = np.asarray(np.meshgrid(*axes)).T.reshape(8,3)
	block_corners += move_vector
	
	corner_vertices = []
	for corner in block_corners:
		corner_vertices.append(mesh.add_vertex(corner))

		
		
	#x+face
	mesh.add_face(corner_vertices[2],corner_vertices[3],corner_vertices[6])
	mesh.add_face(corner_vertices[3],corner_vertices[7],corner_vertices[6])
		
	#x-face
	mesh.add_face(corner_vertices[0],corner_vertices[4],corner_vertices[1])
	mesh.add_face(corner_vertices[1],corner_vertices[4],corner_vertices[5])
		
	#y+face
	mesh.add_face(corner_vertices[3],corner_vertices[1],corner_vertices[5])
	mesh.add_face(corner_vertices[3],corner_vertices[5],corner_vertices[7])
		
	#y-face
	mesh.add_face(corner_vertices[0],corner_vertices[2],corner_vertices[4])
	mesh.add_face(corner_vertices[2],corner_vertices[6],corner_vertices[4])
		
	#z+face
	mesh.add_face(corner_vertices[4],corner_vertices[6],corner_vertices[5])
	mesh.add_face(corner_vertices[6],corner_vertices[7],corner_vertices[5])
		
	#z-face
	mesh.add_face(corner_vertices[2],corner_vertices[0],corner_vertices[1]) 
	mesh.add_face(corner_vertices[2],corner_vertices[1],corner_vertices[3])
		
		
		
	return mesh

def make_deck(wline_points, subdivide = False):
	#clean duplicates
	#wline_points = np.unique(wline_points, axis = 0)
	central_points = np.empty((0,3))
	for point in wline_points:
		if np.isclose(point[1], 0) ==  False:
			central_point = copy.copy(point)
			central_point[1] = 0
			central_points = np.append(central_points, np.expand_dims(central_point, 0), axis = 0)
	
	deck_points = np.append(wline_points, central_points, axis = 0)
	
	
	w_max_index = wline_points.shape[0]
	c_max_index = central_points.shape[0]
	
	#pocetni i zadnji fvi koji nemogu u petlju
	deck_fvi = np.array([[0,0+w_max_index,1],[deck_points.shape[0]-1,w_max_index-1,w_max_index-2]])
	 
	
	for interval_index in range(len(central_points-1)):
		fvi = np.array([[w_max_index,w_max_index+1,1],[w_max_index+1,2,1]]) + interval_index
		deck_fvi = np.append(deck_fvi, fvi, axis = 0)
	
	if subdivide == False:
		return om.TriMesh(deck_points, deck_fvi)
	elif subdivide == True:
		return subdivide_mesh([om.TriMesh(deck_points, deck_fvi)])
	
def move_mesh(mesh, move_vector):
	for vh in mesh.vertices():
		new_point = mesh.points()[vh.idx()] + move_vector
		mesh.set_point(vh, new_point)
	return mesh
	
def is_mesh_closed(mesh):
	if mesh.n_faces()==0:
		return False
	for eh in mesh.edges():		#check if mesh has any boundary edges if not mesh is closed, if yes mesh is open
		if mesh.is_boundary(eh) == True:
			return False
	else:
		return True


def subdivide_mesh(mesh_list, c = 0 ,n = 1): #face po face subdividamo n puta,c je counter
	if c < n:
		new_meshes = []
		for mesh in mesh_list:
			mesh_points =  mesh.points()
			mesh_fvi = mesh.face_vertex_indices().tolist()
			mesh_hei = mesh.face_halfedge_indices().tolist() #lista sa 3 vrijednosti unutra
			face_hevi = mesh.halfedge_vertex_indices().tolist() #heh idx -> vertex dofs	#lista [.........] velika sa slistama od 2 point = points[vindices]
			for i in range(len(mesh_fvi)): #i je idx od fh
				face_points = np.empty((0,3))
				midpoints = np.empty((0,3))
				for j in mesh_hei[i]: #j je idx od heh za halfedgeve na tom faceu /za svaki halfedge handleidx u faceu
					
					hevi = (face_hevi[j])	#tu se vrti
					halfedge_points = mesh_points[hevi] #array 
					face_points = np.append(face_points, np.expand_dims(halfedge_points[0], axis = 0), axis = 0)	# da se zadrzi orijentacija
					midpoint = halfedge_points[0] + (halfedge_points[1] - halfedge_points[0]) * 0.5
					midpoints =  np.append(midpoints, np.expand_dims(midpoint, axis = 0), axis = 0)
				new_mesh = om.TriMesh()
				vhandles = []
				fhandles = []
				for point in np.append(face_points, midpoints, axis = 0):
					vhandles.append(new_mesh.add_vertex(point))
				
				
				fhandles.append(new_mesh.add_face(vhandles[0], vhandles[3], vhandles[5]))
				fhandles.append(new_mesh.add_face(vhandles[3], vhandles[1], vhandles[4]))
				fhandles.append(new_mesh.add_face(vhandles[5], vhandles[3], vhandles[4]))
				fhandles.append(new_mesh.add_face(vhandles[5], vhandles[4], vhandles[2]))
				new_meshes.append(new_mesh)
	
		return subdivide_mesh(new_meshes, c = c + 1, n = n)
	
	else:
		return hard_merge_meshes(mesh_list)
		
		
		
#ovo u numpy:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			
def calc_face_volume(face_points):
	a = face_points[0]
	b = face_points[1]
	c = face_points[2]
	volume = np.abs(np.dot(a, np.cross(b,c))) / 6
	return volume
	
def calc_mesh_volume(mesh):	
	mesh_volume = 0
	for fh in mesh.faces():
		face_normal = mesh.calc_face_normal(fh)
		face_centroid = mesh.calc_face_centroid(fh)
		fvi = mesh.face_vertex_indices()[fh.idx()]
		face_points = mesh.points()[fvi]
		face_volume = calc_face_volume(face_points)
		face_sign = np.sign(np.dot(face_centroid, face_normal))
		mesh_volume += face_volume * face_sign
	return mesh_volume
	
				
if __name__ == "__main__":
				
	import time			
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	
	# p = np.array([[0,0,0],[0,0,10],[0,20,20]])
	# pp = np.array([[[0,0,0],[1,0,0],[0,1,0]],[[0,0,1],[1,0,1],[0,1,1]]])
	# print(dist_p_plane2(p,pp))

	# p = np.array([[1,1,1.999],[-3,-3,-3]])
	# pp = np.array([[[1,0,0],[0,1,0],[0,0,0]],[[0,0,2],[2,0,2],[0,2,2]],[[1,1,-1],[2,1,-1],[1,2,-1]]])
	
	
	# p = np.array([[-3,-3,0],[1,1,1.999999]])
	# pp = np.array([[[1,0,0],[0,1,0],[0,0,0]],[[0,0,2],[2,0,2],[0,2,2]]])
	# print(is_in_triangular_prism2(pp,p, pair_mode = False))
	
	# test problems:
	# bp = np.array([[2, -5.26885, 5.2],[6.781, -5.26885, 5.2]][1:2])
	bp = np.array([[2, -5.26885, 5.2],[6.781, -5.26885, 5.2],[6.781, -5.26885, 5.2],[6.781, -5.26885, 5.2],[2, -5.26885, 5.2],[6.781, -5.26885, 5.2]])
	# tp = np.array([[[5.75445497,-5.20833602,5.2],[8.35713409,-5.3651378,5.2],[8.24153478,-5.97551904,6.25]],[[5.75445497,-5.20833602,5.2],[8.24153478,-5.97551904,6.25],[5.6316307,-5.86084806,6.25]]][0:1])
	tp = np.array([[[5.75445497,-5.20833602,5.2],[8.35713409,-5.3651378,5.2],[8.24153478,-5.97551904,6.25]],[[5.75445497,-5.20833602,5.2],[8.24153478,-5.97551904,6.25],[5.6316307,-5.86084806,6.25]],[[5.75445497,-5.20833602,5.2],[8.35713409,-5.3651378,5.2],[8.24153478,-5.97551904,6.25]],[[5.75445497,-5.20833602,5.2],[8.35713409,-5.3651378,5.2],[8.24153478,-5.97551904,6.25]]])
	a = is_in_triangular_prism2(tp, bp, pair_mode = False)
	print(a[0].keys())
	
	tp = np.append(tp, tp[:,0].reshape(-1,1,3), 1)
	for tri in list(tp):
		ax.plot(tri[:,0],tri[:,1],tri[:,2], "blue")
	ax.scatter(bp[:,0],bp[:,1],bp[:,2], c = "red")
	
	plt.show()
	# import matplotlib.pyplot as plt
	# from mpl_toolkits.mplot3d import Axes3D
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# ax.set_xlabel("$X$")
	# ax.set_ylabel("$Y$")
	# ax.set_zlabel("$Z$")
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	