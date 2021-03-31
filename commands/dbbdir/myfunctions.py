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

#pp je da su suprotne tocke na quadu udaljene za 2 u arrayu	
def quadinter(quads):		#quad je n x 4 x 3 array
	#print(quads)
	bool_same_points = (quads.reshape(-1,1,4,1,3) == quads.reshape(1,-1,1,4,3)).all(-1).any(2)	# where atleast 1 point is same 
	#clean diag:
	a = np.arange(bool_same_points.shape[0])
	diag_indices = (a,a)
	bool_same_points[diag_indices] = False
	# i_row_atleast1 = np.where(bool_same_points.any(-1))   #[0]sa kojim quadovima se sijece ,[1] kojem quadu pripada tocka koja se sijece
	# i_point = np.argmax(bool_same_points[i_row_atleast1], 1)
	# i_opposite_point = (i_point + 2)%4		#suprotni je za 2 indeksa udaljen
	# print(quads, i_row_atleast1, i_opposite_point)
	#print(quads)
	
	#make and clean diagonal bool matrix:
	bool_array = bool_same_points.any(-1)
	#bool_array = bool_array*np.triu(bool_array)		#remove lower tri duplicates, not upper for easier indexing		#if triu the last instance of duplicate array will be removed
	bool_array = np.triu(bool_array)
	i_row1 = np.where(bool_array)   #[1]sa kojim quadovima se sijece ,[0] kojem quadu pripada tocka koja se sijece
	i_row2 = tuple(np.roll(np.asarray(i_row1), 1, 0))  #roll moves elements
	i_point1 = np.argmax(bool_same_points[i_row1], 1)
	i_point2 = np.argmax(bool_same_points[i_row2], 1)
	i_op1 = (i_point1 + 2)%4		#suprotni je za 2 indeksa udaljen
	i_op2 = (i_point2 + 2)%4		#suprotni je za 2 indeksa udaljen
	#sets of quads on same indexes
	q1 = quads[i_row1[1]]
	q2 = quads[i_row2[1]]
	#pairs of opposite points of quads on same indexes:
	op1 = q2[(np.arange(q2.shape[0]), i_op2)]
	op2 = q1[(np.arange(q1.shape[0]), i_op1)]
	bool = (is_in_quad(q1,op1, True) | is_in_quad(q2,op2, True)).flatten()
	#return dofs of intersecting quads:
	return np.asarray(i_row1).T[bool]
	
	# print(is_in_quad(q1,op1, True))
	# print(is_in_quad(q2,op2, True))
	
	# print(bool_same_points)
	# print(bool_same_points[i_row1])
	# print(i_row1,i_row2)
	# print(i_op1,i_op2)
	
	#i_row2 = 
	# i_point = np.argmax(bool_same_points[i_row_atleast1], 1)
	# i_opposite_point = (i_point + 2)%4		#suprotni je za 2 indeksa udaljen
	
	
#	opposite_points = quads[(i_row_atleast1[1], i_opposite_point)]
#	quads_with_intersections = quads[(i_row_atleast1[0],)]
	#print(quads_with_intersections, opposite_points)
	# print("quads:\n" + str(quads_with_intersections))
	# print("points:\n" + str(opposite_points))
#	bool = is_in_quad(quads_with_intersections, opposite_points, True)
#	print(bool.flatten())   remove duplicates
	#print(is_in_quad(quads_with_intersections, opposite_points, True))
	
	
	# bool_atleast1 = bool_same_points.any(-1)
	# bool_atleast1 = bool_atleast1*np.triu(bool_atleast1)		#remove lower tri duplicates, not upper for easier indexing		#if triu the last instance of duplicate array will be removed
	# diag_indices = np.diag_indices(bool_atleast1.shape[0])
	# bool_atleast1[diag_indices] = False
	# i_atleast1 = np.where(bool_atleast1)
	# i_point = np.argmax(bool_same_points[i_atleast1], 1)
	# i_opposite_point = (i_point + 2) % 4
	#i1 = np.append(np.expand_dims(i_atleast1[0],0), ,0)
	
	#opposite_points = quads[i]
	
	
	#print(bool_same_points)
	
def is_in_quad(quads, p, pair_mode = False):
	tri1 = np.delete(quads, 2, 1)
	tri2 = np.delete(quads, 0, 1)
	bool1 = is_in_tri2(tri1, p, pair_mode)
	bool2 = is_in_tri2(tri2, p, pair_mode)
	
	return bool1 | bool2

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
	
	
# def is_in_tri2(fp, p, pair_mode = False):		#rezultat je bool array; axis 0 su pointovi, a axis 1 su trokuti, npr [2][0] je lezi li point 2 u trokutu 0, no ako je pair mode True; racunati ce trokute i pointove na istim indeksima  
	# if pair_mode == False:
		# v0 = fp[:,0,:]
		# u = fp[:,1,:] - v0 
		# v = fp[:,2,:] - v0 
		
		# w = vadjdims(p, v0) - vclone(v0, p)
		
		# uu = (u**2).sum(1)
		# vv = (v**2).sum(1)
		# uv = (u*v).sum(1)
		# wu = (w*u).sum(2)
		# wv = (w*v).sum(2)
		
		# d = uv**2 - uu * vv		#denominator
		# si = (uv*wv - vv*wu) / d
		# ti = (uv*wu - uu*wv) / d
		# siti = si+ti
		# mintol = -0.00001								#povecaj tol ako treba? 
		# maxtol = 1.00001
		# stavi errstate za supressanje nezeljenih warninga
		# with np.errstate(invalid = "ignore"):		#removes warnings of runtime errors encountering nan-s witch works as intended
			# return (si != np.nan) & (ti != np.nan)& (siti != np.nan) & (mintol < si) & (si < maxtol) & (mintol < ti) & (ti < maxtol) & (siti <= maxtol)

		# warnings.filterwarnings('ignore')
	# elif pair_mode == True:
		# if fp.shape[0] != p.shape[0]:
			# raise CustomError("For pair mode triangle points array and points array must have same shape[0]")
	
		
		# v0 = fp[:,0,:]
		# u = fp[:,1,:] - v0 
		# v = fp[:,2,:] - v0 
		
		# w = p - v0
		
		# uu = (u**2).sum(-1)
		# vv = (v**2).sum(-1)
		# uv = (u*v).sum(-1)
		# wu = (w*u).sum(-1)
		# wv = (w*v).sum(-1)		

		# d = uv**2 - uu * vv		#denominator
		# si = (uv*wv - vv*wu) / d
		# ti = (uv*wu - uu*wv) / d
		# siti = si+ti
		# mintol = -0.00001								#povecaj tol ako treba? 
		# maxtol = 1.00001
		# stavi errstate za supressanje nezeljenih warninga
		# with np.errstate(invalid = "ignore"):		#removes warnings of runtime errors encountering nan-s witch works as intended
			# return (si != np.nan) & (ti != np.nan)& (siti != np.nan) & (mintol < si) & (si < maxtol) & (mintol < ti) & (ti < maxtol) & (siti <= maxtol)
		
		
		# print(w)
	
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

	
def unique_close2(a, axis = 0):		#a is nxm		#for axis >1 use only when all instances have a duplicate; else they won't be deleted to conserve array shape 
	#b = np.expand_dims(a,axis + 1) 	#b is  nx1xm for broadcasting
	#b = np.expand_dims(a,axis = (1,3))
	#b = a
	#a = a.reshape((5,1,3,3))
	#b = b.reshape(5,3,1,3)
	#print(b.shape)
	a1 = np.expand_dims(a, axis + 1)		#osiguraba multidim broadcasting
	a2 = np.expand_dims(a, axis)
	c = np.isclose(a1,a2).all(tuple(range(axis + 2, len(a1.shape))))		#bool array dim reduction .all(extra axes)
	#c = c*np.tril(c)		#remove upper tri duplicates		#if triu the last instance of duplicate array will be removed
	c = np.tril(c)		#remove upper tri duplicates		#if triu the last instance of duplicate array will be removed
	
	diag_indices = np.diag_indices(c.shape[axis])
	
	if axis > 0:	#no need for multidim dofs #!!!!!!!!!!!jos treba prosiriti za dim > 1
		i = a.shape[0]
		x1 = np.expand_dims(np.concatenate([np.arange(i)]*c.shape[axis]).reshape(-1,i).T.flatten(), 0)		#axis0 array dofs
		x2 = np.concatenate([np.asarray(diag_indices)]*i, 1)
		diag_indices = tuple(np.concatenate((x1,x2),0))
	c[diag_indices] = False
	counts = c.sum(axis)
	c = ~(counts.astype(bool))
	return (a[c])		#
	
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



#----------------------------------------
#intersection algorithm table
#http://www.realtimerendering.com/intersections.html


def is_in_tri(p, v0, u, v): #u, v triangle plane vectors from v0, p is point to test #works only if in same plane
		#plane_points = np.array([v0, v0+u, v0+v])
		#if np.isclose(dist_p_plane(p, plane_points),0):	#point must be on plane
		w = p - v0 
		uu = np.dot(u,u)
		vv = np.dot(v,v)
		uv = np.dot(u,v)
		wu = np.dot(w,u) 
		wv = np.dot(w,v)
		denominator = uv**2 - uu * vv
		si = (uv*wv - vv*wu) / denominator
		#print(si)
		
		mintol = -0.00001								#povecaj tol ako treba? 
		maxtol = 1.00001
		if si < mintol or si > maxtol:			#jos ubrzat sa 1.000000001 uvijetom?
			return False
		ti = (uv*wu - uu*wv) / denominator
		#print(ti)
		if ti < mintol or (si+ti) > maxtol:
			return False
		return True

		
		#if (si < 0.0 and np.isclose(si,0.0) == False) or (si > 1.0 and np.isclose(si, 1.0) == False):			#jos ubrzat sa 1.000000001 uvijetom?
		#	return False
		#ti = (uv*wu - uu*wv) / denominator
		##print(ti)
		#if (ti < 0.0 and np.isclose(ti,0.0) == False) or ((si+ti) > 1.0 and np.isclose((si+ti),1.0) == False):
		#	return False
		#return True
	#else:
		#	return False
			
#http://geomalgorithms.com/a05-_intersect-1.html			
def line_line_inter(l1, l2):			#general intersection; returns touple with parameters for l2 equation to get l1 points, and in case of only 1 intersection; that point
	u = l1[1] - l1[0]
	v = l2[1] - l2[0]
	#i = np.where(v != 0)[0][0] 		#index where denominator v is not 0 to avoid error   if both u and v [i] are 0 paralel check will be True regardless of other indexes           
	#if np.all((u[i]*v - u*v[i]) == 0.0):					#paralell check
	if np.all((u[0]*v - u*v[0]) == 0.0):					#paralell check
		w = l1[0] - l2[0]
		if np.all(w[0]*v - w*v[0] == 0.0):				#coincidance check
			z = l1[1] - l2[0]
			i = np.where(v != 0)[0][0]		#premijesteno
			t0 = w[i]/v[i]				#sto ako je v 0? index np where v =! 0?
			t1 = z[i]/v[i]
			return (t0, t1)
		else:
			return (None)
	else:				#lines are not paralell
		for i in range(u.shape[0]):		#find non paralel projection of rd
			up = np.delete(u, i)	#remove i coordinate
			vp = np.delete(v, i)
			if np.any((up[0]*vp - up*vp[0]) != 0.0):	#if rd projections are not paralel
				index = i
				break
		else:	#ako su svi paralelni?
			return (None)
		
		w = l1[0] - l2[0]		#P0 - Q0
		wp = np.delete(w, index)
		denominator = up[0] * vp[1] - up[1] * vp[0]
		sI = -(vp[1]*wp[0] - vp[0]*wp[1]) / denominator		#za l1
		tI = (up[0]*wp[1] - up[1]*wp[0]) / denominator 		#za l2
		Pi1 = l1[0] + sI * u
		Pi2 = l2[0] + tI * v
		if np.allclose(Pi1, Pi2):
			return (sI, tI, Pi1)
		else:
			return (None)
	
def segment_inter(s1,s2): #segments are paralel and colinear
	line_vector = s1[1] - s1[0]
	i = np.where(line_vector != 0)[0][0] #coordinate index where both segments are not 0
	
	s1min = np.min(s1[:,i])
	s1max = np.max(s1[:,i])
	s2min = np.min(s2[:,i])
	s2max = np.max(s2[:,i])
	
	
	if (s1min <= s2min <= s1max) or (s1min <= s2max <= s1max) or (s2min <= s1min <= s2max) or (s2min <= s1max <= s2max):
		u = np.append(s1, s2, 0) #unity array
		mini = np.where(u[:,i] == np.min(u[:,i]) )[0][0] #first row index with min valid value
		maxi = np.where(u[:,i] == np.max(u[:,i]) )[0][0] #first row index with max valid value
		return np.delete(u, (mini, maxi), 0)  #return unity with max and min points deleted
	else:
		return None
	
	
def do_segments_cross(l1,l2):	#assume they are in same plane and are not paralel
	u = l1[1] - l1[0]
	v = l2[1] - l2[0]
	for i in range(u.shape[0]):		#find non paralel projection of rd
		up = np.delete(u, i)	#remove i coordinate
		vp = np.delete(v, i)
		if np.any((up[0]*vp - up*vp[0]) != 0.0):	#if rd projections are not paralel
			index = i
			break
	else:
		return False			#added, check if ok later
	
	w = l1[0] - l2[0]		#P0 - Q0
	wp = np.delete(w, index)
	denominator = up[0] * vp[1] - up[1] * vp[0]
	sI = -(vp[1]*wp[0] - vp[0]*wp[1]) / denominator		#za l1
	tI = (up[0]*wp[1] - up[1]*wp[0]) / denominator 		#za l2
	Pi1 = l1[0] + sI * u
	Pi2 = l2[0] + tI * v
	#print(Pi1, Pi2, sI, tI)
	if np.allclose(Pi1, Pi2) and 0 <= sI <= 1 and 0 <= tI <= 1:
		return True
	else:
		return False

	
	
	
#http://geomalgorithms.com/a04-_planes.html#Distance-Point-to-Plane
def dist_p_plane(p, plane_points):
	u = plane_points[1] - plane_points[0]
	v = plane_points[2] - plane_points[0]
	v0 = plane_points[0]
	n = np.cross(u,v)
	d = -np.dot(n, v0)		
	dn = np.sum(n**2)**0.5	
	return (np.sum(n*p)+d) / dn	

def is_p_outside_face(p, face_points):	#p is outisde if is on + side of normal, inside if on face or - side of normal
	if dist_p_plane(p, face_points) > 0.0:
		return True
	else:
		return False
	
	
#http://geomalgorithms.com/a06-_intersect-2.html
def line_plane_inter(ro,rd, plane_points): #ro line segment start, r1 line segment end, v0 plane origin point
	u = plane_points[1] - plane_points[0]
	v = plane_points[2] - plane_points[0]
	n = np.cross(u,v)
	if np.all(n == 0.0):	#degenerate triangle
		return(None,)
	a = np.dot(n,(plane_points[0] - ro))
	if np.isclose(a, 0.0):	#if ro is on plane
		return (ro, 0.0 ,u ,v)	#intersection is ray origin and t = 0
	
	denominator = np.dot(n,rd)
	if np.isclose(denominator, 0.0):	#if True, Ray is paralel to plane and on plane condition was checked
		return (None,)
	t = a / denominator		#ray and plane are not paralel
	pi = ro + rd*t
	return (pi, t ,u ,v)
		#if return is inf line and plane are paralel if return >= 1 intersection is outside segment 

#http://geomalgorithms.com/a06-_intersect-2.html
def line_tri_inter(ro, rd, triangle_points): 				#ro = ray origin, rd = ray direction
	inter_data = line_plane_inter(ro,rd, triangle_points)
	if inter_data[0] is not None:	#intersection with tri plane exists
		pi = inter_data[0]									#intersection point
		t = inter_data[1]
		u = inter_data[2]
		v = inter_data[3]
		#old_con = is_inside_triangle(pi, triangle_points)
		new_con = is_in_tri(pi, triangle_points[0], u, v)
		#if is_inside_triangle(pi, triangle_points):
		#if is_in_tri(pi, triangle_points[0], u, v):		#AKO ZAMIJENIM SA STARIM DOBRO FUNKCIONIRA?
		#if old_con != new_con:
		#	print("\nerror at point: " + str(ro))
		#	print("\nerror at direction: " + str(rd))
		#	print("error at tri_points: " + str(triangle_points))
		if new_con:
			return (pi, t)
		else:
			return (None,)
	else:
		return (None,)


#http://geomalgorithms.com/a06-_intersect-2.html			
def tri_plane_inter(plane_points, triangle_points):
	in_p = np.empty((0,3))
	out_p = np.empty((0,3))
	on_p = np.empty((0,3))			#dodaj uvijet paralenosti?
	for p in triangle_points:			#ako je sporo probaj tolist()
		dist = dist_p_plane(p, plane_points)
		if dist > 0.0:
			out_p = np.append(out_p, np.expand_dims(p,0),0)
		elif dist < 0.0:
			in_p = np.append(in_p, np.expand_dims(p,0),0)
		elif dist == 0.0:
			on_p = np.append(on_p, np.expand_dims(p,0),0)
		#else: #if triangle is degenerate: add cross product = 0 outside 
		#	return (None)
		
	inlen = in_p.shape[0]
	outlen = out_p.shape[0]
	onlen = on_p.shape[0]
	
	if inlen == 3 or outlen == 3 or onlen == 3:	#all points are on one side of plane	#dodaj uvijet paralenosti?
		return (None)
	
	elif onlen == 2 and (inlen == 1 or outlen == 1):
		return (on_p)
	
	elif inlen == 2 and outlen == 1 :														#all points are not on one side and all points are not on plane
		inter = np.empty((0,3))
		v = in_p - out_p #vector array	#out_p is just one point
		for rd in list(v):
			i = line_plane_inter(out_p[0],rd,plane_points)[0]
			inter = np.append(inter, np.expand_dims(i,0),0) 
		return (inter)
		
	elif inlen == 1 and outlen == 2:
		inter = np.empty((0,3))
		v = out_p - in_p #vector array	#in_p is just one point
		for rd in list(v):
			i = line_plane_inter(in_p[0],rd,plane_points)[0]
			inter = np.append(inter, np.expand_dims(i,0),0)  
		return (inter)

	elif inlen == 1 and outlen == 1 and onlen == 1:
		inter = on_p
		rd = (out_p - in_p)[0]
		i = line_plane_inter(in_p[0],rd,plane_points)[0]
		inter = np.append(inter, np.expand_dims(i,0),0)
		return (inter)
		
	elif onlen == 1 and (inlen == 2 or outlen == 2):		#if 1 or 2 points are on plane and other points are all on one side
		#return on_p unneeded for face cut algorithm
		return (None)
	
	
	else:
		return(None)

#http://geomalgorithms.com/a06-_intersect-2.html		
def tri_tri_inter(tri1, tri2):	#tri1 is block triangle!
	s1 = tri_plane_inter(tri1, tri2)
	s2 = tri_plane_inter(tri2, tri1)
	if s1 is not None and s2 is not None and s1.shape[0] == 2 and s2.shape[0] == 2:		#triangles can intersect with plane in only 1 point and 3 points, in those cases return None, because this algorithm is for face cutting and only 1 or 3 intersections are not needed 
		segment = segment_inter(s1,s2)		#data1[1] is block face inside points #face that is to be cut
		#print("yoyo: " + str(segment)+ str("\n"))
		if (segment is None) or np.allclose(segment[0], segment[1]):		#ako se trokuti sijeku u samo 1 tocki trokut nam ne treba
			return None
		else:
			return segment
	else:
		return None

def get_neigbour_faces(fh_idx_list, mesh_ffi, n = 2): 		#rewrite sa setom		# ima problem sa dodavanjem nepovezanih faceva, neznam jeli to do pogreske u formi, novi uvijet sa is_boundary? 
	#return_list = copy.copy(fh_idx_list)
	#delta_fh_idx = copy.copy(fh_idx_list) #provjeravaj samo za addane	
	
	#for x in range(n):
	#	memory = []
	#	for fh_idx in delta_fh_idx:
	#		near_idx_list = mesh_ffi[fh_idx]
	#		try:
	#			near_idx_list.remove(-1)
	#		except:
	#			pass
	#		for idx in near_idx_list:
	#			if idx not in return_list:
	#				return_list.append(idx)
	#				memory.append(idx)
	#	delta_fh_idx = copy.copy(memory)
	
	
	fh_set = set(fh_idx_list)
	for x in range(n):
		new_fh = set()
		for fh_idx in fh_set:
			delta = mesh_ffi[fh_idx]
			try:
				delta.remove(-1)
			except:
				pass
			new_fh.update(delta)
		fh_set.update(new_fh)
	return list(fh_set)
	
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
	
	
	
	
def get_faces_near_block(block_mesh, form_mesh, block_dims, block_position):  #n => algorithm repeat n times
	#1) koji facevi su blizu blocka
	form_mesh_vfi = form_mesh.vertex_face_indices().tolist()
	form_mesh_points = form_mesh.points().tolist()
	xmin = block_position[0] 
	xmax = block_position[0] + block_dims[0]
	ymin = block_position[1] 
	ymax = block_position[1] + block_dims[1]
	zmin = block_position[2]
	zmax = block_position[2] + block_dims[2]
	
	near_fh_idx_list = []
	form_inside_points = np.empty((0,3))
	
	#trazi fh_idx od svih faceva koji imaju tocku u projekciji sa blokm na xz ravninu
	for vh_idx in range(len(form_mesh_points)):
		point = form_mesh_points[vh_idx]
		if (xmin <= point[0] <= xmax) and (ymin <= point[1] <= ymax) and (zmin <= point[2] <= zmax): #ako je point izmedju projekcije
			near_fh_idx_list += form_mesh_vfi[vh_idx] 
			form_inside_points = np.append(form_inside_points, np.expand_dims(point,0), 0)
			
	#micanje duplikata i -1 u listi
	near_fh_idx_list = set(near_fh_idx_list)
	near_fh_idx_list = list(near_fh_idx_list)
	try:
		near_fh_idx_list.remove(-1)
	except:
		pass

	return (near_fh_idx_list, form_inside_points)

def get_faces_near_block2(block_mesh, form_mesh, block_dims, block_position):
	form_mesh_vfi = form_mesh.vertex_face_indices()
	form_mesh_points = form_mesh.points()
	xmin = block_position[0] 
	xmax = block_position[0] + block_dims[0]
	ymin = block_position[1] 
	ymax = block_position[1] + block_dims[1]
	zmin = block_position[2]
	zmax = block_position[2] + block_dims[2]
	
	bool = (((xmin <= form_mesh_points[:,0]) & (form_mesh_points[:,0] <= xmax)) & ((ymin <= form_mesh_points[:,1]) & (form_mesh_points[:,1] <= ymax)) & ((zmin <= form_mesh_points[:,2]) & (form_mesh_points[:,2] <= zmax)))
	form_inside_vh_idx = np.where(bool)[0]		#[0] because return tuple
	
	form_inside_fh_idx = form_mesh_vfi[form_inside_vh_idx].flatten()
	form_inside_fh_idx = np.unique(np.delete(form_inside_fh_idx, np.where(form_inside_fh_idx == -1)))	#clean -1 and remove duplicates
	
	return (form_inside_fh_idx, form_inside_vh_idx)
	
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

def are_all_points_colinear(points):
	vectors = np.delete((points - points[0]),0, 0)		
	cross_products = np.cross(np.delete(vectors,0,0), vectors[0])		#v1xv2,v1xv3,v1xv3.....
	if np.isclose(cross_products, 0).all():
		return True
	else:
		return False
	
	
	
def cut_mesh2(block_mesh, form_mesh, block_dims, block_position):
	
	# import matplotlib.pyplot as plt
	# from mpl_toolkits.mplot3d import Axes3D
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	


	#get form_inside_vh_idx and extract form segment:
	form_data = get_faces_near_block2(block_mesh, form_mesh, block_dims, block_position)
	form_fh_idx_to_check = get_neigbour_faces2(form_data[0], form_mesh,block_dims, block_position ,n = 2)
	form_inside_vh_idx = form_data[1]
	data1 = extract_mesh2(form_fh_idx_to_check, form_mesh, form_inside_vh_idx)		#duplikati u originalnoj formi!
	# print(data1[1])
	data2 = clean_mesh(data1[0], data1[1])
	form_segment_mesh = data2[0]
	form_segment_inside_vh_idx = data2[1]
	
	# print(form_segment_inside_vh_idx)
	
	#get block_inside_vh_idx:
	block_points = block_mesh.points()
	form_segment_fvi = form_segment_mesh.face_vertex_indices()
	form_segment_points = form_segment_mesh.points()
	block_inside_vh_idx = np.where((dist_p_plane2(block_points, form_segment_points[form_segment_fvi]) <= (0 + 1e-08)).all(-1))[0]
	if block_inside_vh_idx.shape[0] == 8:	#if block is completely in 
		return block_mesh
	
	#precalc block and segment normals:
	block_tpoints = block_points[block_mesh.face_vertex_indices()]
	block_normals = np.cross(block_tpoints[:,1] - block_tpoints[:,0], block_tpoints[:,2] - block_tpoints[:,0])
	block_normals = block_normals/((block_normals**2).sum(-1)**0.5).reshape(-1,1)
	
	form_segment_tpoints = form_segment_points[form_segment_fvi] 
	form_segment_normals = np.cross(form_segment_tpoints[:,1] - form_segment_tpoints[:,0], form_segment_tpoints[:,2] - form_segment_tpoints[:,0])
	form_segment_normals = form_segment_normals/((form_segment_normals**2).sum(-1)**0.5).reshape(-1,1)
	
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
	
	block_fvi = block_mesh.face_vertex_indices()
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
		points = unique_close2(block_data_dict[block_fh])
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
		
	
	
	#make new clean mesh function(with option to sync vh and fh), new stitch
	#either stitch face dosen't work as intended or form mesh has duplicates! both
	# print("block n points:")
	# print(hard_merge_meshes2(block_merge_mesh_list).points().shape[0])

	# print("form n points:")
	# print(hard_merge_meshes2(form_segment_merge_mesh_list).points().shape[0])	
	
	# for mesh in block_merge_mesh_list:
		# tp = mesh.points()[mesh.face_vertex_indices()]
		# v0 = tp[:,0]
		# centroids = tp.sum(1)/3
		# normals = np.cross(tp[:,1] - tp[:,0], tp[:,2] - tp[:,0])
		# normals = normals/((normals**2).sum(-1)**0.5).reshape(-1,1)
		# normals = np.append(centroids.reshape(-1,1,3), (normals + centroids).reshape(-1,1,3), 1)
		# tp = np.append(tp, tp[:,0].reshape(-1,1,3), 1)
		# for tri in list(tp):
			# ax.plot(tri[:,0],tri[:,1],tri[:,2], "blue")
		# ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2], c = "blue")
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
				
	final_mesh = hard_merge_meshes2(block_merge_mesh_list + form_segment_merge_mesh_list)
	#final merged mesh plot:
	# points = final_mesh.points()
	# evi = final_mesh.edge_vertex_indices()
	# for eh in final_mesh.edges():
		# if final_mesh.is_boundary(eh):
			# ep = points[evi[eh.idx()]]
			# ax.plot(ep[:,0],ep[:,1],ep[:,2], "red")
			
			
	# plt.show()
	
	return final_mesh
	
	
	
	# form_inside_points = form_mesh.points()[form_inside_vh_idx]
	# form_seg_inside_points = form_segment_mesh.points()[form_segment_inside_vh_idx] + np.full(3, 0.01)
	# ax.scatter(form_seg_inside_points[:,0],form_seg_inside_points[:,1],form_seg_inside_points[:,2], c ="blue")
	# ax.scatter(form_inside_points[:,0],form_inside_points[:,1],form_inside_points[:,2], c = "green")
	
	
	# fvi = form_segment_mesh.face_vertex_indices()
	# points = form_segment_mesh.points()[fvi]
	# for trip in points:
		# trip = np.append(trip, np.expand_dims(trip[0],0),0)
		# ax.plot(trip[:,0],trip[:,1],trip[:,2], "blue")
	
	# fvi = block_mesh.face_vertex_indices()
	# points = block_mesh.points()[fvi]
	
	# for trip in points:
		# trip = np.append(trip, np.expand_dims(trip[0],0),0)
		# ax.plot(trip[:,0],trip[:,1],trip[:,2], "red")
	
	
	# plt.show()
	

	
def cut_mesh(block_mesh, form_mesh, block_dims, block_position):			#unfinished
	start = time.time()
	uncut_block = copy.copy(block_mesh)
	#get form data(faces to check and inside vh):
	form_data = get_faces_near_block(block_mesh, form_mesh, block_dims, block_position)
	#get more faces for safety:
	mesh_ffi = form_mesh.face_face_indices().tolist()
	form_fh_idx_to_check = get_neigbour_faces(form_data[0], mesh_ffi, n = 2)
	form_inside_points = form_data[1]
	form_segment_mesh = hard_merge_meshes([extract_mesh(form_fh_idx_to_check, form_mesh)])
	uncut_segment = copy.copy(form_segment_mesh)
	
	block_fvi_list = list(block_mesh.face_vertex_indices())
	block_points = block_mesh.points()
	
	form_fvi_list = list(form_segment_mesh.face_vertex_indices())
	form_points = form_segment_mesh.points()	
	segment_inside_vh_idx_list = sync_vh_idx(form_inside_points, form_points)
	
	block_bad_fh_idx = []
	form_bad_fh_idx = []
	block_cut_faces_mesh_list = []
	form_segment_cut_faces_mesh_list = []
	block_inside_vh_idx_list = []
	block_outside_vh_idx_list = []			#vh idx to be deleted 
	block_face_data = dict()			#key is fh_idx, [0] is inside_vh_idx, [1] is unordered sets if intersection points
	form_face_data = dict()
	
	#get block inside vh_idx list: 													#ovo dobro radi
	j = np.array([0,1,0])
	for vh_idx in range(block_points.shape[0]):
		point = block_points[vh_idx]
		for face_fh_idx in range(len(form_fvi_list)):
			face_fvi = form_fvi_list[face_fh_idx]
			face_points = form_points[face_fvi]
			inter_data = line_tri_inter(point, j, face_points)	#[0] is None if no intersection
			if inter_data[0] is not None:	#there is an intersection
				pi_y = abs(inter_data[0][1])		#y coordinate of intersection
				if abs(point[1]) <= pi_y:
					block_inside_vh_idx_list.append(vh_idx)
					break
				else:
					block_outside_vh_idx_list.append(vh_idx)		
					break
		else:					#if intersection was none for all faces then point is outside by x axis
			block_outside_vh_idx_list.append(vh_idx)		
	
	#make form face division dict here so no repetitition occurs in for loop below
	for form_fh_idx in range(len(form_fvi_list)):										#dobro
		form_fvi = form_fvi_list[form_fh_idx]
		form_face_points = form_points[form_fvi]
		if np.allclose(np.cross(form_face_points[1]-form_face_points[0], form_face_points[2]-form_face_points[0]), 0.0) == False:	#if face is not degenerate
			form_inside_vh_idx = []			#treba li outside set?
			for form_vertex_idx in list(form_fvi):
				if form_vertex_idx in segment_inside_vh_idx_list:
					form_inside_vh_idx.append(form_vertex_idx)
				else:
					#treba li outside set?
					pass	
			#face_normal = form_mesh.calc_face_normal(form_mesh.face_handle(form_fh_idx))     
			face_normal = np.cross(form_face_points[2]-form_face_points[0],form_face_points[1]-form_face_points[0])
			form_face_data[form_fh_idx] = [form_inside_vh_idx, np.empty((0,3)), face_normal]	#update za sve bez obzira ima li inside ili nema
		
	#block face point division and updating intersections
	for block_fh_idx in range(len(block_fvi_list)):
		block_fvi = list(block_fvi_list[block_fh_idx])
		block_face_points = list(block_points[block_fvi])
		block_inside_vh_idx = []
		for block_vertex_idx in block_fvi:	#division of block vh_idx by inside and outside
			if block_vertex_idx in block_inside_vh_idx_list:
				block_inside_vh_idx.append(block_vertex_idx)
		if len(block_inside_vh_idx) == 1 or len(block_inside_vh_idx) == 2:	#if there are 1 or 2 inside points for this face:
			face_normal = np.sign(block_mesh.calc_face_normal(block_mesh.face_handle(block_fh_idx)))
			block_face_data[block_fh_idx] = [block_inside_vh_idx, np.empty((0,3)), face_normal]  #[1] is intersection pairs 
			#update dict with empty intersect pairs
			#get intersection:
			i = 0
			for form_fh_idx in list(form_face_data.keys()):		#for all valid faces in faces to check
				form_fvi = form_fvi_list[form_fh_idx]
				form_face_points = list(form_points[form_fvi])
				#intersections:
				intersection_points =  tri_tri_inter(block_face_points, form_face_points)
				if intersection_points is not None:			#intersection must not be none; will allways have 2 intersection points			
					block_face_data[block_fh_idx][1] = np.append(block_face_data[block_fh_idx][1],intersection_points ,0)
					inlen = len(form_face_data[form_fh_idx][0])
					if inlen == 1 or inlen == 2:
						#points_to_add = np.empty((0,3))
						#inside_vh_idx = form_face_data[form_fh_idx][0]
						#inside_points = form_points[inside_vh_idx]
						#for intersection_point in list(intersection_points):
						#	print(inside_points, intersection_point)
						#	if np.equal(inside_points, intersection_point).all(1).any() == False:	# if intersection point is same as an inside point
						#		points_to_add = np.append(points_to_add, np.expand_dims(intersection_point,0),0)
						form_face_data[form_fh_idx][1] = np.append(form_face_data[form_fh_idx][1], intersection_points,0)
					
#				if i == 9 and block_fh_idx == 1:		#(i,j)  (8,0)  (9,1)krivo
#					print("ping")
#								
#					import matplotlib.pyplot as plt
#					from mpl_toolkits.mplot3d import Axes3D
#					fig = plt.figure()
#					ax = fig.add_subplot(111, projection='3d')
#					ax.set_xlabel("$X$")
#					ax.set_ylabel("$Y$")
#					ax.set_zlabel("$Z$")
#					plot_fh_3d([form_fh_idx], form_fvi_list, form_points, ax, "blue")
#					#plot_fh_3d(list(range(len(block_fvi_list))), block_fvi_list, block_points, ax, "red")
#					plot_fh_3d([block_fh_idx], block_fvi_list, block_points, ax, "red")
#					data = (1, block_face_data[block_fh_idx])
#					#for data in list(block_face_data.items()):
#					inside_points = block_points[data[1][0]]
#					#intersection_points = data[1][1]			#change unique close tolerances if needed
					#ax.scatter(intersection_points[:,0],intersection_points[:,1],intersection_points[:,2], "orange")		#tri_tri se sijece sa krivim mjestima
					#ax.scatter(inside_points[:,0],inside_points[:,1],inside_points[:,2], "purple")			#inside pointovi su dobri
					
					#for fh in form_fh_idx_to_check:
					#	fvi = form_fvi_list[fh]
					#	points = form_points[fvi]
					#	tri = np.append(points, np.expand_dims(points[0], 0), 0)
					#	ax.plot(tri[:,0],tri[:,1],tri[:,2], "blue")
					#ax.plot(tri1[:,0],tri1[:,1],tri1[:,2], "red")
					#ax.plot(tri2[:,0],tri2[:,1],tri2[:,2], "green")
#					if intersection_points is not None:
#						ax.scatter(intersection_points[:,0],intersection_points[:,1],intersection_points[:,2], c = "orange")
#					s1 = tri_plane_inter(block_face_points, form_face_points)
#					s2 = tri_plane_inter(form_face_points, block_face_points)
#					ax.scatter(s1[:,0],s1[:,1],s1[:,2],  c = "red")			#inside pointovi su dobri
#					ax.scatter(s2[:,0],s2[:,1],s2[:,2],  c = "blue")
#					print(intersection_points,s1,s2)
#					print(segment_inter(s1,s2))
					
#					line_vector = s1[1] - s1[0]
#					i = np.where(line_vector != 0)[0][0] #coordinate index where both segments are not 0
					
#					s1min = np.min(s1[:,i])
#					s1max = np.max(s1[:,i])
#					s2min = np.min(s2[:,i])
#					s2max = np.max(s2[:,i])
					
#					print(s1min <= s2min <= s1max, s1min <= s2max <= s1max, s2min <= s1min <= s2max, s2min <= s1max <= s2max)
	#				if (s1[0][i] <= s2[0][i] <= s1[1][i]) or (s1[0][i] <= s2[1][i] <= s1[1][i]) or (s2[0][i] <= s1[0][i] <= s2[1][i]) or (s2[0][i] <= s1[1][i] <= s2[1][i]):
	#					u = np.append(s1, s2, 0) #unity array
	#					mini = np.where(u[:,i] == np.min(u[:,i]) )[0][0] #first row index with min valid value
	#					maxi = np.where(u[:,i] == np.max(u[:,i]) )[0][0] #first row index with max valid value
	#					print(np.delete(u, (mini, maxi), 0))  #return unity with max and min points deleted
	#				else:
	#					print("error") 
					
					
					#print(s1, s2)
					
					
					#print(block_face_points)
					#print(form_face_points)
					#print(intersection_points)
					
#					plt.show()
								
#				i+=1
				
	#make cut block meshes:				#imamo prazne intersection pointove u dictu a imaju inside pointove!
	for data in list(block_face_data.items()):
		#fh_idx = data[0]
		#fvi = block_fvi_list [fh_idx]
		inside_points = block_points[data[1][0]]
		intersection_points = data[1][1]			#change unique close tolerances if needed
#		if intersection_points.shape[0] == 0 and (inside_points.shape[0] == 1 or inside_points.shape[0] == 2):			#postoje pointovi sa 0 intersecta i 1 ili 2 inside tocke! ---> tri tri ne radi dobro!!!!!!!!!!!!!!!
		#print(inside_points)
		#print(intersection_points)
		intersection_points = sort_by_dist(unique_close(intersection_points))
		#print(intersection_points)
		cut_mesh = stitch_face(inside_points, intersection_points)
		normal = np.sign(cut_mesh.calc_face_normal(cut_mesh.face_handle(0)))	 #all are in same plane 
		if np.array_equal(normal, data[1][2]) == False: 	#normals and axes are paralell so sign is enough to check for paralel
			cut_mesh = flip_mesh_face_orientation(cut_mesh)
		block_cut_faces_mesh_list.append(cut_mesh)		#stitch ne radi dobro!!!!	probaj jos povecat tolerancije stitcha

	#make cut form segment meshes:
	for data in list(form_face_data.items()):
		intersection_points = data[1][1]
		inside_vh_idx = data[1][0]
		if intersection_points.shape[0] > 0:
			fh_idx = data[0]
			inside_points = form_points[inside_vh_idx]
			intersection_points = sort_by_dist(unique_close(intersection_points))
			#print(inside_points, intersection_points)
			cut_mesh = stitch_face(inside_points, intersection_points)
			if cut_mesh.face_vertex_indices().shape[0] > 0: #if there are faces in mesh
				n1 = data[1][2]
				points = cut_mesh.points()[cut_mesh.face_vertex_indices()[0]]
				#n2_p = cut_mesh.calc_face_normal(cut_mesh.face_handle(0))	 #all are in same plane
				n2 = np.cross(points[2]-points[0],points[1]-points[0])
				#print(cut_mesh.face_vertex_indices())
				#print(cut_mesh.points()[cut_mesh.face_vertex_indices()[0]])
				#print(n1_p,n2_p,np.dot(n1_p,n2_p))

				
				#i = np.where(n2_p != 0)[0][0]
				#print(i)
				if np.dot(n1,n2) < 0.0: # if normals are not in same direction		
					cut_mesh = flip_mesh_face_orientation(cut_mesh)
				form_segment_cut_faces_mesh_list.append(cut_mesh)		#stitch ne radi dobro!!!!	probaj jos povecat tolerancije stitcha
		#elif len(inside_vh_idx) == 0:
		#	fh_idx = data[0]
	
	for bad_vh_idx in block_outside_vh_idx_list:
		block_mesh.delete_vertex(block_mesh.vertex_handle(bad_vh_idx), False)
	block_mesh.garbage_collection()
	block_cut_faces_mesh_list.insert(0, block_mesh)
	#block_cut_faces_mesh_list = [block_cut_faces_mesh_list[0]]
	
	for vh_idx in range(form_points.shape[0]):
		if vh_idx not in segment_inside_vh_idx_list:
			vh = form_segment_mesh.vertex_handle(vh_idx)
			form_segment_mesh.delete_vertex(vh, True)
	form_segment_mesh.garbage_collection()
	form_segment_cut_faces_mesh_list.insert(0, form_segment_mesh)
	#form_segment_cut_faces_mesh_list = [form_segment_mesh]

	
	
	
	
	
	
#	print (form_segment_mesh.points())
#	print(form_segment_mesh.face_vertex_indices())
#	print(np.isclose(form_segment_mesh.points()[0],form_segment_mesh.points()[1]))
#	print(form_segment_mesh.n_faces())
#	for fvi in form_segment_mesh.face_vertex_indices():
#		points = form_segment_mesh.points()[fvi]
#		print(np.cross(points[2]-points[0],points[1]-points[0]))
#	#complex edgevi; neki facevi se sijeku na topu i bottomu blocka
#	import time			
#	import matplotlib.pyplot as plt
#	from mpl_toolkits.mplot3d import Axes3D
#	fig = plt.figure()
#	ax = fig.add_subplot(111, projection='3d')
#	ax.set_xlabel("$X$")
#	ax.set_ylabel("$Y$")
#	ax.set_zlabel("$Z$")

	
	
	
	
	#	block_fh_to_plot = list(range(12))		#error at (8,9), [10,11] 10
#	plot_fh_3d(form_fh_idx_to_check, form_fvi_list, form_points, ax, "blue")
	#plot_fh_3d(list(range(len(block_fvi_list))), block_fvi_list, block_points, ax, "red")
#	plot_fh_3d(block_fh_to_plot, block_fvi_list, block_points, ax, "red")
#	for data in list(block_face_data.items()):
#		if data[0] in block_fh_to_plot:
#			inside_points = block_points[data[1][0]]
#			intersection_points = data[1][1]			#change unique close tolerances if needed
#			intersection_points = sort_by_dist(unique_close(intersection_points))
			#print(intersection_points)
			#print(len((intersection_points)))
#			ax.scatter(intersection_points[:,0],intersection_points[:,1],intersection_points[:,2], c = "purple")		#tri_tri se sijece sa krivim mjestima
#			ax.scatter(inside_points[:,0],inside_points[:,1],inside_points[:,2], c = "orange")			#inside pointovi su dobri
#			print("inside_points:")
#			print(inside_points)
#			print("intersection_point:")
#			print(intersection_points)
#			cut_block_mesh = stitch_face(inside_points, intersection_points)
#			plot_fh_3d(list(range(len(list(cut_block_mesh.face_vertex_indices())))), list(cut_block_mesh.face_vertex_indices()), cut_block_mesh.points(), ax, "purple")
			#for fh in form_fh_idx_to_check:
			#	fvi = form_fvi_list[fh]
			#	points = form_points[fvi]
			#	tri = np.append(points, np.expand_dims(points[0], 0), 0)
			#	ax.plot(tri[:,0],tri[:,1],tri[:,2], "blue")
			#ax.plot(tri1[:,0],tri1[:,1],tri1[:,2], "red")
			#ax.plot(tri2[:,0],tri2[:,1],tri2[:,2], "green")
			#ax.scatter(intersection_points[:,0],intersection_points[:,1],intersection_points[:,2], "orange")

	cut_block_mesh = hard_merge_meshes(block_cut_faces_mesh_list + form_segment_cut_faces_mesh_list)		#soft merge potvrduje da svi facevi tamo i postoje na pravim mjestima, na baisic paineru sam edito
	#plot_fh_3d(list(range(len(list(uncut_segment.face_vertex_indices())))), list(uncut_segment.face_vertex_indices()), uncut_segment.points(), ax, "blue")
	#plot_fh_3d(list(range(len(list(uncut_block.face_vertex_indices())))), list(uncut_block.face_vertex_indices()), uncut_block.points(), ax, "Red")
	
	#for mesh in form_segment_cut_faces_mesh_list:
	#	plot_fh_3d(list(range(len(list(mesh.face_vertex_indices())))), list(mesh.face_vertex_indices()), mesh.points(), ax, "purple")
#	i = [0]
	#for x in i:
	#	print("flies")
	#	print(form_segment_cut_faces_mesh_list[x].points())
		#plot_fh_3d(list(range(len(list(form_segment_cut_faces_mesh_list[x].face_vertex_indices())))), list(form_segment_cut_faces_mesh_list[x].face_vertex_indices()), form_segment_cut_faces_mesh_list[x].points(), ax, "purple")
	
#	i = [6]
	#for x in i:
	#	print("block")
	#	print(block_cut_faces_mesh_list[x].points())
	#	plot_fh_3d(list(range(len(list(block_cut_faces_mesh_list[x].face_vertex_indices())))), list(block_cut_faces_mesh_list[x].face_vertex_indices()), block_cut_faces_mesh_list[x].points(), ax, "purple")
		









#		plot_fh_3d(list(range(len(list(cut_block_mesh.face_vertex_indices())))), list(cut_block_mesh.face_vertex_indices()), cut_block_mesh.points(), ax, "blue")
	
#	fh_to_plot = [16,15]
#	for fvi in list(cut_block_mesh.face_vertex_indices()):
#		points = cut_block_mesh.points()[fvi]
#		if np.allclose(np.cross(points[2]-points[0],points[1]-points[0]), 0):
#			points = np.append(points,np.expand_dims(points[0],0),0)
#			ax.plot(points[:,0],points[:,1],points[:,2], c = "red")
#	
#	ax.scatter(cut_block_mesh.points()[-1][0],cut_block_mesh.points()[-1][1],cut_block_mesh.points()[-1][2], c = "green")
#	ax.scatter(cut_block_mesh.points()[5][0],cut_block_mesh.points()[5][1],cut_block_mesh.points()[5][2], c = "green")
#	
#	fvi_list = cut_block_mesh.face_vertex_indices()
#	print(fvi_list)
#	for fh_idx in fh_to_plot:
#		fvi = fvi_list[fh_idx]
#		points = cut_block_mesh.points()[fvi]
#		#print(fvi,points)
#		points = np.append(points,np.expand_dims(points[0],0),0)
#		ax.plot(points[:,0],points[:,1],points[:,2], c = "red")
#	
#	print(cut_block_mesh.face_face_indices()[16])
		










		#print("aa")
	#print(cut_block_mesh.points())
	#print(np.isclose(cut_block_mesh.points()[-1],cut_block_mesh.points()[5]))
	#print(cut_block_mesh.points()[-1]-cut_block_mesh.points()[5])
	
	#to_plot = 1
	#i=0
	#for data in list(form_face_data.items()):
	#	inter = data[1][1]
	#	if inter.shape[0] != 0:
	#		ax.scatter(inter[:,0],inter[:,1],inter[:,2], c = "orange")
	#		i+=1
	#	if i == to_plot:
	#		break
	#print(form_inside_points)
	#ax.scatter(form_inside_points[:,0],form_inside_points[:,1],form_inside_points[:,2], c = "green")
	#plt.show()

	
#	print("1,5")
	#print(block_cut_faces_mesh_list[3].points())
	#print(block_cut_faces_mesh_list[3].face_vertex_indices())
	#test_mesh_points = np.array([[50,0,15.4],[50,6.71557609,15.4],[50,6.8226125,14.5812865]]) 
	#test_mesh_points = np.array([[1.0,0.0,0.0],[2.0,0.0,2.0],[3.0,0.0,1.0]]) 
	#test_mesh_fvi = np.array([[0,1,2]])
	#print(test_mesh_points, test_mesh_fvi)
	#test_mesh = om.TriMesh(test_mesh_points, test_mesh_fvi)
	#print(test_mesh.face_normals(),test_mesh.face_normals()[0],test_mesh.face_normals()[1],test_mesh.face_normals()[2])
	#print(np.cross(test_mesh_points[2]-test_mesh_points[0],test_mesh_points[1]-test_mesh_points[0]))
	#print(block_cut_faces_mesh_list[5].face_normals())
					#hard merge ne radi dobro!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! (1,5)
#	print(block_cut_faces_mesh_list[1].points(), block_cut_faces_mesh_list[1].face_vertex_indices())
#	print(block_cut_faces_mesh_list[5].points(), block_cut_faces_mesh_list[5].face_vertex_indices())
#	print("merge")
	#cut_block_mesh = soft_merge_meshes(block_cut_faces_mesh_list + [form_segment_mesh])
	#cut_block_mesh = test_mesh
#	test_tri = np.array([[7,0,12.4],[0,25,12.4],[7,25,12.4]]) #na ovom trekutu nema nikakvih intersectiona tri_plane inter je los
#	inter_array = np.empty((0,3))
#	for form_fh_idx in range(len(form_fvi_list)):
#		fvi = form_fvi_list[form_fh_idx]
#		face_points = form_points[fvi]
#		if np.allclose(np.cross(face_points[1]-face_points[0], face_points[2] - face_points[0]), 0) == False:
#			inter = tri_tri_inter(test_tri, face_points)
#			if inter is not None:
#				inter_array = np.append(inter_array, inter)
	
	#print(inter_array)		
				#	if inter_array.shape[0] == 0:
#		print("ping")

	#print(len(block_face_data))
	
	print("mesh cut time: " + str(time.time()-start))
	#print("new_cut volume:")
	#print(calc_mesh_volume(cut_block_mesh)) 	#old fit mesh volume = 144.22083786629366 new is 144.5123765
	return cut_block_mesh
	
def plot_fh_3d(mesh_fh_list, mesh_fvi_list, points_array, ax, color = "green"):
	for fh in mesh_fh_list:
		fvi = mesh_fvi_list[fh]
		points = points_array[fvi]
		tri = np.append(points, np.expand_dims(points[0], 0), 0)
		ax.plot(tri[:,0],tri[:,1],tri[:,2], color)



	
def sort_pairs(unsorted_pairs):
	unsorted_pairs = unsorted_pairs
	sorted_points = np.empty((0,3))
	start_point_data = np.unique(unsorted_pairs.reshape(-1,3), axis = 0, return_counts = True)
	start_point = start_point_data[0][np.where(start_point_data[1] == 1)][0]	#prvi point koji se pojavljuje samo jedan put
	sorted_points = np.append(sorted_points, np.expand_dims(start_point, 0), 0)
	print("unsorted_pairs: " + str(unsorted_pairs) + "\n")
	print("start_point: " + str(start_point) + "\n")
	print("sortedpairs: " + str(sorted_points) + "\n") 
	for i in range(unsorted_pairs.shape[0]):
		index_data = np.where(np.isclose(unsorted_pairs, start_point).all(2) == True)
		print("unsorted_pairs: " + str(unsorted_pairs) + "\n")
		print("start_point: " + str(start_point) + "\n")
		print("index_data: " + str(index_data) + "\n")
		print("sortedpairs: " + str(sorted_points) + "\n") 
		next_point = unsorted_pairs[index_data[0][0]][(index_data[1]+1)%2]
		sorted_points = np.append(sorted_points, next_point, 0)
		start_point = next_point
		unsorted_pairs = np.delete(unsorted_pairs, index_data[0][0], 0)
	return sorted_points
	
def unique_close(array, return_counts = False):		#returns unique close values of array
	sorted_array = np.empty((0,3))
	if return_counts == False:
		for point in list(array):
			condition = np.isclose(array, point).all(1)
			if condition.any():								#if any point is close to any other in array 
				index_list = np.where(condition == True)[0]
				sorted_array = np.append(sorted_array, np.expand_dims(point,0), 0)
				array = np.delete(array, index_list, 0)
		
		return sorted_array
	else:
		count_list = np.empty((0), dtype = "int64")
		for point in list(array):
			condition = np.isclose(array, point).all(1)
			if condition.any():								#if any point is close to any other in array 
				index_list = np.where(condition == True)[0]
				sorted_array = np.append(sorted_array, np.expand_dims(point,0), 0)
				count_list = np.append(count_list, int(np.array([index_list.shape[0]])))
				array = np.delete(array, index_list, 0)
		
		return (sorted_array, count_list)


		
	# np.isclose(a, np.array([0,1,2])).all(1)  #check for close values
def sort_by_dist(points, sort_by_min_dist = True):  # need to clean all close values
	aritm_middle = np.sum(points, axis = 0)/points.shape[0]
	d = aritm_middle - points
	distance = np.sum(d**2, axis = 1)**0.5	
	i = np.array(np.where(distance == distance.max())).flatten()[0]	#sto ako su 2 tocke jednako udaljene?	[0] da odabere samo 1
	start_point = points[i]
	sorted_array_shape = list(points.shape)				#if points have n dims
	sorted_array_shape[0] = 0
	sorted_array = np.empty(sorted_array_shape)
	if sort_by_min_dist == True:
		for p in range(points.shape[0]):
			#print(start_point)
			#print("-------------")
			#print(points)
			d = start_point - points
			distance = np.sum(d**2, axis = 1)**0.5
			i = np.array(np.where(distance == distance.min())).flatten()[0]	#sto ako su 2 tocke jednako udaljene?	[0] da odabere samo 1
			#print("index" + str(i))
			start_point = points[i]
			points = np.delete(points, i, axis = 0)
			#print("star point"+str(start_point))
			sorted_array = np.append(sorted_array, np.expand_dims(start_point, axis = 0), axis = 0)
	
	elif sort_by_min_dist == False:		#trazimo po max duljini
		for p in range(points.shape[0]):
			#print(start_point)
			#print("-------------")
			#print(points)
			d = start_point - points
			distance = np.sum(d**2, axis = 1)**0.5
			i = np.array(np.where(distance == distance.max())).flatten()[0]	#sto ako su 2 tocke jednako udaljene?	[0] da odabere samo 1
			#print("index" + str(i))
			start_point = points[i]
			points = np.delete(points, i, axis = 0)
			#print("star point"+str(start_point))
			sorted_array = np.append(sorted_array, np.expand_dims(start_point, axis = 0), axis = 0)
		
	return sorted_array

def extract_mesh(fh_idx_to_extract, mesh): #extracts faces from mash and makes a new mesh from them
	extracted_fvi = mesh.face_vertex_indices()[fh_idx_to_extract]
	extracted_mesh = om.TriMesh(mesh.points(), extracted_fvi)
	delete_isolated_vertices(extracted_mesh)
	return extracted_mesh

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
	
	
def sync_vh_idx(points_to_sync, cut_mesh_points):		#syncs vh_idx of points	
	vh_idx_list = []
	for point in list(points_to_sync):
		vh_idx_list += np.where(np.equal(cut_mesh_points, point).all(1) == True)[0].tolist()
	return vh_idx_list
		
	
	
def stitch_face(inside_points, intersection_points):		
	tlen = intersection_points.shape[0]
	mlen = tlen + inside_points.shape[0]
	a = np.arange(2,mlen)
	b = np.full(a.shape[0],0)
	c = a - 1
	mesh_fvi = np.column_stack((a,b,c))
	
	if inside_points.shape[0] == 2:
		p1 = (inside_points[0], intersection_points[0])
		p2 = (inside_points[1], intersection_points[-1])
		if do_segments_cross(p1,p2) == False:
			mesh_fvi[0] = np.array([1,0,(mlen-1)])
	
	
	mesh_points = np.append(inside_points, intersection_points, 0)
	mesh = om.TriMesh(mesh_points, mesh_fvi)
	for fh_idx in range(mesh_fvi.shape[0]):
		fvi = mesh_fvi[fh_idx]
		face_points = mesh_points[fvi]
		normal = np.cross(face_points[2]-face_points[0],face_points[1]-face_points[0])
		if np.isclose(normal,0).all():	#if face is degenerate; delete it
			fh = mesh.face_handle(fh_idx)
			mesh.delete_face(fh, True)
	mesh.garbage_collection()
	#print(mesh.face_vertex_indices(), mesh.points())
	return mesh
	
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
	
	
	
#	#print(inside_points)
	#if inside_points.shape[0] == 2:
#	#intersection_points = sort_by_dist(intersection_points)	#rewrite to iterate over list
#	tlen = intersection_points.shape[0]
#	mlen = tlen + inside_points.shape[0]
#	a = np.arange(2,mlen)
#	b = np.full(a.shape[0],0)
#	c = a - 1
#	mesh_fvi = np.column_stack((a,b,c))

	
#	if inside_points.shape[0] == 2:
#		p1 = (inside_points[0], intersection_points[0])
#		p2 = (inside_points[1], intersection_points[-1])
#		print(do_segments_cross(p1,p2))
#		if do_segments_cross(p1,p2):
#			intersection_points = intersection_points[::-1]
#			mesh_fvi[0] = np.array([mlen-1,0,1])
#		else:
#			inside_points = inside_points[::-1]
#	mesh_points = np.append(inside_points, intersection_points, 0)
	#print(mesh_points)
	#print("\n")
	#print(mesh_fvi)
#	print(mesh_points, mesh_fvi)
#	return om.TriMesh(mesh_points, mesh_fvi)
	#elif E.shape[0] == 1:
	#	pass

	
				
	
#-------------------------------------------------------------------------------		
		
def array_where_equal(a,b, bool = True):		#ako hocemo stupce samo stavi a.T , ako hocemo i gdje je razlicito stavimo bool = False
	i_array = np.empty(0, dtype = "int64")
	for i in range(a.shape[0]):
		if np.array_equal(a[i], b) == bool:
			i_array = np.append(i_array, i)
			
	return i_array


def flip_mesh_face_orientation(mesh):
	flipped_fvi = np.flip(mesh.face_vertex_indices(), axis = 1)
	return om.TriMesh(mesh.points(), flipped_fvi)
	

def clean_mesh(mesh, vh_idx_to_sync = None):
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
	

def delete_degenerate_faces(mesh, vh_idx_to_sync_list = None):
	if vh_idx_to_sync_list is None:
		pass
	else:
		# normals
		pass
def hard_merge_meshes2(meshes, vh_idx_to_sync = None):			#vh_idx_to_sync_list is list with numpy arrays
	if vh_idx_to_sync is None:
		merged_mesh = soft_merge_meshes(meshes)
		merged_mesh = soft_merge_meshes(meshes)
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
	

	
#close i sa degenerate face checkom! ili u hullformu promijeniti sa unique wlinove?

#radi complex edgeve!! jesu li im pointovi close?
def hard_merge_meshes(meshes):   #meshes je lista sa meshevima
	merged_mesh = soft_merge_meshes(meshes)
	#data = np.unique(merged_mesh.points(), return_counts = True, axis = 0)
	data = unique_close(merged_mesh.points(), return_counts = True)	
	unique_points = data[0]
	duplicate_counter = data[1]
	points_with_duplicates = unique_points[np.where(duplicate_counter > 1)]
	new_vh = []
	for dpoint in list(points_with_duplicates):
		new_vh.append(merged_mesh.add_vertex(dpoint))
			
	bad_fh_list = []
	new_fvi_list = []
	
	
	merged_mesh_fvi = merged_mesh.face_vertex_indices().tolist()
	merged_mesh_points = merged_mesh.points()
	for i in range(len(merged_mesh_fvi)):	#trazimo bad fh i mijenjamo njihov fvi u novi
		fvi = np.asarray(merged_mesh_fvi[i])		
		face_points = merged_mesh_points[fvi]
		new_fvi = copy.copy(fvi)
		for nvh in new_vh:		#trazi jeli novi vh i face imaju istu tocku
			new_point = merged_mesh_points[nvh.idx()] 
			
			#new_fvi[array_where_equal(face_points, new_point)] = nvh.idx()
			new_fvi[np.where(np.equal(face_points, new_point).all(1))] = nvh.idx()
		if np.array_equal(fvi, new_fvi) == False:		#ako originalni i novi fvi nisu isti dodaje novi fvi u listu
			fh = merged_mesh.face_handle(i)
			bad_fh_list.append(fh)
			new_fvi_list.append(new_fvi)
			
			
	for bad_fh in bad_fh_list:					#delete bad faces:
		merged_mesh.delete_face(bad_fh, False)  	#false da ne deletea izolirane vertexe
	
	merged_mesh.garbage_collection()
		
	
	for new_fvi in new_fvi_list:				#retriangularizacija sa novim fvi	
		new_face_vhandles = []
		for vi in new_fvi:
			new_vh = merged_mesh.vertex_handle(vi)
			new_face_vhandles.append(new_vh)
		
		merged_mesh.add_face(new_face_vhandles)
			
	delete_isolated_vertices(merged_mesh)
			
	return merged_mesh	
							 
	

def delete_isolated_vertices(mesh): 
	mesh_vertex_face_indices = list(mesh.vertex_face_indices()) #kod vertex_face_indices izolirani su oni kojima je svima -1 (arrajevi moraju biti svi istevelicine pa je null -1)
	for vh_idx in range(mesh.points().shape[0]):
		neighbouring_faces_fh_idx = mesh_vertex_face_indices[vh_idx]
		if np.all(neighbouring_faces_fh_idx == -1):
			vh = mesh.vertex_handle(vh_idx)
			mesh.delete_vertex(vh)
	mesh.garbage_collection()

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

		
def triangle_surface(triangle_points):
	AB = triangle_points[1]-triangle_points[0]
	AC = triangle_points[2]-triangle_points[0]
	surface = np.linalg.norm(np.cross(AB, AC))/2
	return surface


	
def is_inside_triangle(point, triangle_points):
	ABC = triangle_surface(triangle_points)
	Ai = np.empty((0))
	for i in range(triangle_points.shape[0]):
		points = copy.copy(triangle_points)
		points[i] = point
		Ai = np.append(Ai, triangle_surface(points))
	
	if np.isclose(np.sum(Ai), ABC) == True:
		return True
	else:
		return False

	
	
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
	wline_points = np.unique(wline_points, axis = 0) 
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
		
	
def get_intersection(radius, vector, face_points):
	face_radius = face_points[0]
	face_vector1 = face_points[1] - face_points[0]
	face_vector2 = face_points[2] - face_points[0]
	
	#face_radius, face_vector1, face_vector2,
	
	radius_matrix = (radius - face_radius).T
	vector_matrix = np.empty((3,3))
	vector_matrix[:,0] = face_vector1
	vector_matrix[:,1] = face_vector2
	vector_matrix[:,2] = -vector
		
	try:
		edge_parameter = np.linalg.solve(vector_matrix, radius_matrix)[2]
	except:
		return (None, None)
	
	intersection_point = radius + (vector * edge_parameter)
	if is_inside_triangle(intersection_point, face_points) == True:
		return (intersection_point, edge_parameter)
	else:
		return (None, edge_parameter)



def is_mesh_closed(mesh):	
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
		
		
		

def plot3D(points_list):
	scatter_points = []
	colors = ["red","green","yellow","blue","orange","purple","black","cyan","magneta"]
	fig=plt.figure()
	axes = plt.axes(projection='3d')
	axes.set_xlabel("x")
	axes.set_ylabel("y")
	axes.set_zlabel("z")
	
	for i in range(len(points_list)):
		points = points_list[i]
		color = colors[i % (len(colors)-1)]
		x = points[:,0]
		y = points[:,1]
		z = points[:,2]
		scatter_points.append(get_scatter(axes, x, y, z, color))

	#points1 = axes.scatter3D(points[:,0],points[:,1],points[:,2], color = "green"); 
	#points2 = axes.scatter3D(supertr_points[:,0],supertr_points[:,1],supertr_points[:,2], color = "red"); 
	#point_center = axes.scatter3D(points_centroid[0],points_centroid[1],points_centroid[2], color = "blue");
	#supertr_center = axes.scatter3D(supertr_centroid[0],supertr_centroid[1],supertr_centroid[2], color = "black");		
	plt.show()
	
   
			
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
	
def make_block_csv_file():
	block_dims = np.array([1,1,1])
	mesh = make_block(block_dims)
	mesh = subdivide_mesh([mesh], n = 1)
	points = mesh.points().tolist()
	fvi = mesh.face_vertex_indices().tolist()

	with open("unit_block_points.csv", "w", newline = "") as csv_file:
		csv_writer = csv.writer(csv_file)
		for point in points:
			csv_writer.writerow([point[0],point[1],point[2]])
	
	with open("unit_block_fvi.csv", "w", newline = "") as csv_file:
		csv_writer = csv.writer(csv_file)
		for f in fvi:
			csv_writer.writerow([f[0],f[1],f[2]])
	
def make_form_points_as_csv(form_points):
	with open("C:\\Users\\Tomislav\\Desktop\\Py_Prog\\form_points.csv", "w", newline = "") as csv_file:
		csv_writer = csv.writer(csv_file)
		for point in form_points:
			csv_writer.writerow(point)


	
def	make_block_from_unit_csv(block_dims = np.array([1,1,1]), move_vector = np.array([0,0,0]), path = ""):
	with open(path + "unit_block_points.csv", "_r", newline ="") as csv_file:
		csv_reader = csv.reader(csv_file)
		points = np.asarray([line for line in csv_reader]).astype(float)
		
	with open(path + "unit_block_fvi.csv", "_r", newline ="") as csv_file:
		csv_reader = csv.reader(csv_file)
		fvi = np.asarray([line for line in csv_reader]).astype(int)
	

	return om.TriMesh(points * block_dims + move_vector, fvi)
	

	
				
if __name__ == "__main__":
				
	import time			
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	

	
	# tri1 = np.array([[0,0,0],[-1,0,0],[0,1,0]])
	# tri2 = np.array([[0,0,0],[1,0,0],[0,1,0]])
	# tri1_fvi = np.array([[0,1,2]])
	# tri2_fvi = np.array([[0,1,2]])
	# tri1_mesh = om.TriMesh(tri1, tri1_fvi)
	# tri2_mesh = om.TriMesh(tri2, tri2_fvi)
	
	# print(tri1_mesh.face_normals())
	# print(tri2_mesh.face_normals())
	# print(tri1_mesh.calc_face_normal(tri1_mesh.face_handle(0)))
	# combined_mesh = hard_merge_meshes([tri1_mesh,tri2_mesh])		#ne mijenja orijentaciju!!
	# print(combined_mesh.n_faces())
	# print("done.")
	
	
	#replace tests
	# array = np.array([[1,2,1],[3,8,2],[9,2,5]])
	# a = np.array([8,9,1])
	# replace(array, a, np.arange(a.shape[0]))
	
	#sort by angle tests:
	#o_normal = np.array([0,0,1])
	#points = np.array([[3,0,0],[2,3,0],[2,-3,0],[-2,0,0]])
	#print(sort_by_angle(points, o_normal))
	
	# points = np.array([[9,1.5,7.5],[9,4.987,8.25],[9,4.845,7.5],[9,1.5,10]])
	# points = np.array([[2,0,0],[0,-2,0],[-2,0,0],[0,2,0]])
	#original_normal = np.array([1,0,0])
	# original_normal = np.array([0,0,1])
	# sorted_points = sort_by_angle(points, original_normal)
	# print(points)
	# print(sorted_points)
	
	# for mesh in block_cut_meshes_list[0:1]:
		# tp = mesh.points()[mesh.face_vertex_indices()]
		# print(mesh.face_vertex_indices())
		# print(mesh.points())
		# v0 = tp[:,0]
		# centroids = tp.sum(1)/3
		# normals = np.cross(tp[:,1] - tp[:,0], tp[:,2] - tp[:,0])
		# print(normals[0],normals[1])
		# normals = normals/((normals**2).sum(-1)**0.5).reshape(-1,1)
		# normals = np.append(centroids.reshape(-1,1,3), (normals + centroids).reshape(-1,1,3), 1)
		# tp = np.append(tp, tp[:,0].reshape(-1,1,3), 1)
		# for tri in list(tp):
			# ax.plot(tri[:,0],tri[:,1],tri[:,2], "blue")
		# ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2], c = "blue")
		# for n in list(normals):
			# ax.plot(n[:,0],n[:,1],n[:,2], "blue")
	
	
	
	#tri_tri_test:
#	block_points = np.array([[57,0,12.4],[57,25,12.4],[57,0,15.4]])
#	form_face_points = np.array([[55.20122969, 6.86506929, 13.9],[55.18058696, 6.67028586, 15.4],[57.77088044, 6.61292697, 15.4]])
#	intersection_points_from_cut = np.array([[57, 6.68856821, 14.95000863],[57, 6.73519584,14.5917765]]) 
#	print(tri_tri_inter(block_points, form_face_points))
#	print(tri_tri_inter(block_points, form_face_points))
	
	# form_points = np.array([[[0.16342165, 0, 8.75],[0,4.95,10],[0.16342165,4.7095108,8.75]],[[0.16342165,0,8.75],[0,0,10],[0,4.95,10]],[[0.16342165,4.7095108,8.75],[1.31578947,5.00639569,10],[1.47060998,4.77489939,8.75]],[[0.16342165,4.7095108,8.75],[0,4.95,10],[1.31578947,5.00639569,10]]])
	# block_points = np.array([[[-2,1.5,10],[9,1.5,10],[-2,6.5,10]],[[9,1.5,10],[9,6.5,10],[-2,6.5,10]]])
	# form_points = np.array([[[0.16342165,0,8.75],[0,0,10],[0,4.95,10]]])
	# form_points = np.array([[[6.69936332,5.00104038,8.75],[6.57894737,5.2014511,10],[7.89473684,5.24258705,10]]])
	# block_points = np.array([[[9,1.5,10],[9,6.5,10],[-2,6.5,10]]])
	
	# print(dist_tri_p_plane2(form_points, block_points))
	# print(dist_tri_p_plane2(block_points, form_points))
	# data1 = tri_plane_inter2(form_points, block_points)
	# data2 = tri_plane_inter2(block_points, form_points)
	# s1 = data1[0]
	# s2 = data2[0]
	
	
	# for p in list(s1):
		# ax.scatter(p[:,:,0],p[:,:,1],p[:,:,2], "blue")

	# for p in list(s2):
		# ax.scatter(p[:,:,0],p[:,:,1],p[:,:,2], "blue")
		
	# print(form_points[:,:,2])
	
	# for tri in list(form_points):
		# tri = np.append(tri, tri[0:1], 0)
		# ax.plot(tri[:,0],tri[:,1],tri[:,2], "blue")
	
	# for tri in list(block_points):
		# tri = np.append(tri, tri[0:1], 0)
		# ax.plot(tri[:,0],tri[:,1],tri[:,2], "blue")
	
	# data = tri_tri_inter2(block_points, form_points)	

	# for p in list(data[0]):
		# ax.scatter(p[:,0],p[:,1],p[:,2], "blue")
	
		
	# plt.show()
	
	
	#point = np.array([2,2,2])
	#point_vector_j = np.array([1,1,1])
	#face_points = np.array([[10,0,0],[0,10,0],[0,0,10]])
#	m = 10000
	#start_new = time.time()
	#for i in range(m):
	#	inter = line_tri_inter(point, point_vector_j, face_points)
	#print("new_exe_time: " + str(time.time()-start_new))
#
#	start_new = time.time()
#	for i in range(m):
#		inter = get_intersection(point, point_vector_j, face_points)
#	print("old_exe_time: " + str(time.time()-start_new))
	
	#is_in_tri test		#old = 5s new = 0.181			#ako koristim stari is_inside_triangle line_tri_inter dobro funkcionira
	#ro = np.array([0,3.125,13.9])	#ovo nije dobro!!!
	#rd = np.array([1.0,0.0,0.0])
	#tri_points = np.array([[0.58831793,0,13.9],[0.78442391,5.9889293, 15.4],[0.58831793,6.17923163,13.9]])
	#u = tri_points[1] - tri_points[0]
	#v = tri_points[2] - tri_points[0]
	#pi = line_plane_inter(ro, rd, tri_points)[0]
	#print(pi)
	#print(is_in_tri(pi, tri_points[0], u , v))
	#print(is_inside_triangle(pi,tri_points))
	#start_new = time.time()
	#for i in range(m):
	#	a = is_in_tri(p, v0, u, v)
	#print("new_exe_time: " + str(time.time()-start_new))
	#
	#tri_points = np.array([[v0],[v0+u],[v0+v]])
	#start_new = time.time()
	#for i in range(m):
	#	a = is_inside_triangle(p, tri_points)
	#print("old_exe_time: " + str(time.time()-start_new))
	
	#block_face_points = np.array([[7,6.25,13.9],[7,9.375,13.9],[7,6.25,14.275]])
	#form_face_points = np.array([[5.78954762, 6.32534163, 13.9],[8.55530434, 6.20031354, 15.4],[8.39016247, 6.39230443, 13.9]])
	#tri1 = np.append(block_face_points, np.expand_dims(block_face_points[0], 0), 0)
	#tri2 = np.append(form_face_points, np.expand_dims(form_face_points[0], 0), 0)
	#intersection_points = tri_tri_inter(block_face_points, form_face_points)[0]
	#print(intersection_points)
	
	#ray = np.array([ro,ro+rd])
	#tri = np.array([tri_points[0], tri_points[1], tri_points[2], tri_points[0]])
	#ax.plot(tri1[:,0],tri1[:,1],tri1[:,2], "red")
	#ax.plot(tri2[:,0],tri2[:,1],tri2[:,2], "green")
	#ax.scatter(intersection_points[:,0],intersection_points[:,1],intersection_points[:,2], "orange")
	#plt.show()
	
	#array = np.array([[1,2,3],[1,2,3],[4,5,3],[1,2,5],[0,0,0],[0,0,0.1],[1,2,3.00000001]])
	#print(unique_close(array))
	
	#E = np.array([[1,0,0],[0,0,0]])
	#T_array = np.array([[0,0,3],[1,0,3],[2,0,3],[4,0,3]])
	#print(stitch_face(E,T_array))
	
	#hard merge meshes 2 tests:
	# points1 = np.array([[1,0,0],[0,1,0],[0,0,0],[0.5,0,0]])
	# fvi1 = np.array([[0,1,2]])
	# mesh1 = om.TriMesh(points1, fvi1)
	
	# points2 = np.array([[-1,0,0],[0,0,0],[0,1,0],[-0.5,0.5,0]])
	# fvi2 = np.array([[0,1,2],[0,2,3]])
	# mesh2 = om.TriMesh(points2, fvi2)
	
	# points3 = np.array([[1,0,0],[1,1,0],[0,1,0],[3,3,3]])
	# fvi3 = np.array([[0,1,2]])
	# mesh3 = om.TriMesh(points3, fvi3)
	
	# points4 = np.array([[-1,0,0],[0,1,0],[-1,1,0]])
	# fvi4 = np.array([[0,1,2]])
	# mesh4 = om.TriMesh(points4, fvi4)
	
	# vh_idx_to_sync = [np.array([0,1,2]), np.array([0,1,2]), np.array([3])]
	# meshes = [mesh1, mesh2, mesh3]
	
	# data = hard_merge_meshes2(meshes, vh_idx_to_sync)
	# print(data[1])
	# print(data[0].points(),data[0].face_vertex_indices())
	# mesh = data[0]
	# mesh = hard_merge_meshes2(meshes)
	
	# print(mesh.points(), mesh.face_vertex_indices())
	
	#clean mesh tests:
	
	# points1 = np.array([[1,0,0],[0,1,0],[0,0,0],[1,0,0],[-1,0,0]])
	# fvi1 = np.array([[0,1,2],[3,1,0],[2,1,4]])
	# mesh1 = om.TriMesh(points1, fvi1)
	# mesh =  soft_merge_meshes([mesh1, mesh2])
	
	# print(clean_mesh(mesh1))
	
	
	#delete isolated_verices2 test:, clean test:
	# points1 = np.array([[1,0,0],[2,2,2],[0,1,0],[0,0,0],[1,0,0],[-1,0,0],[3,3,3]])
	# fvi1 = np.array([[0,2,3],[4,2,0],[3,2,5]])
	# mesh1 = om.TriMesh(points1, fvi1)
	
	# data = clean_mesh(mesh1)
	
	# print(data.points(),data.face_vertex_indices())
	#are points colinear tests:
	# points = np.array([[0,0,0],[2,0,0],[4,0,0]])
	# print(are_all_points_colinear(points))
	
	#stitch face2 tests:
	
	# points = np.array([[1,1,0],[1,-1,0],[-1,1,0],[-2,-2,0]])
	# normal = np.array([0,0,1])
	# mesh = stitch_face2(points, normal)
	
	# print(mesh.points(), mesh.face_vertex_indices())
	
	
	#line_plane_inter3 debug:
	# tp = np.array([[[11,6.5,7.5],[0,6.5,7.5],[0,6.5,10]]])
	# pp = np.array([[[0.16342165,0,8.75],[0,4.95,10],[0.16342165,4.7095108,8.75]]])
	# set = line_plane_inter3(tp,pp)
	# print(set)
	
	# tp = np.append(tp, tp[:,0].reshape(-1,1,3), 1)
	# pp = np.append(pp, pp[:,0].reshape(-1,1,3), 1)
	# for tri in list(tp):
		# ax.plot(tri[:,0],tri[:,1],tri[:,2], "green")

	# for tri in list(pp):
		# ax.plot(tri[:,0],tri[:,1],tri[:,2], "blue")
	# ax.scatter(set[:,:,0],set[:,:,1],set[:,:,2], c="red")
	
	a = np.array([[[1,2,3],[4,5,2],[4,5,2]],[[8,8,8],[2,2,2],[8,8,8]]])
	b = np.full((3,2,3),0)
	print(a,b)
	c = unique_close2(a,axis = 1).reshape(-1,2,3)
	print(c)
	b[[True,False,True]] = c
	print(b)
	plt.show()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	