# coding=utf-8
import numpy as np
import json
import os

from .BasicCalculaters import calculatervector,myvectorizationNorm,dist2linevectorization
from .offsetNodelist import *

def LatticeCellSDF(IF,rs,resolution):
    resu = np.zeros([resolution + 1, resolution + 1, resolution + 1], dtype=float)
    #k可以理解为精度参数，k越大判断点是否在杆内会越精确
    k = 256
    for i in range(len(IF)):
        #IF[i]:qm中所有点到第i条线段的距离
        #rs[i]:晶格杆半径
        #tempM:qm中所有点到第i条杆外壁（减去杆半径）的距离,tempM<=0就说明点在杆里
        tempM = IF[i] - rs[i]
        #np.exp(-k * tempM):e^(-128*tempM)
        #resu(41,41,41):累加得到点到每根杆的距离相关的公式
        resu = resu + np.exp(-k * tempM)
    FF = np.log(resu) / k
    return FF

def p2voxel(mtype, parameters, resolution = 40):
    resolution = resolution -1
    strutlist = getStrutlistFromJson(mtype)
    nodelist = get_nodelist(parameters)

    rs = [parameters['R1'],parameters['R2'],parameters['R3']] * 64
    VN_T,Edge = symm48(nodelist,strutlist)

    h = 1.0 / resolution
    #np.mgrid(start:end:point)
    #0,h,2h,3h......nh<=1
    x, y, z = np.mgrid[0:1 + h / 2:h, 0:1 + h / 2:h, 0:1 + h / 2:h]
    qm = np.zeros([resolution + 1, resolution + 1, resolution + 1, 3], dtype=float)
    qm[:, :, :, 0] = x
    qm[:, :, :, 1] = y
    qm[:, :, :, 2] = z
    #qm是40x40x40方格的点（所以是41x41x41）

    IF = []
    # construct implicit function per truss
    for sdf in range(Edge.shape[0]):
        temp_ff = np.zeros([resolution + 1, resolution + 1, resolution + 1], dtype=float)
        sp = VN_T[Edge[sdf, 0], :]  # start point
        ep = VN_T[Edge[sdf, 1], :]  # end point

        rim = calculatervector(sp, ep, qm)
        sp_normm = myvectorizationNorm(np.subtract(qm, sp), 2)
        ep_normm = myvectorizationNorm(np.subtract(qm, ep), 2)
        distm = dist2linevectorization(sp, ep, qm)

        Index = rim <= 0
        temp_ff[Index] = sp_normm[Index]

        Index = rim >= 1
        temp_ff[Index] = ep_normm[Index]

        Index = ((rim > 0) & (rim < 1))
        temp_ff[Index] = distm[Index]

        #temp_ff:qm上每个点到线段的距离

        IF.append(temp_ff)

    FF = LatticeCellSDF(IF,rs,resolution=resolution)
    FF[FF>=0]=1
    FF[FF<0]=0

    voxel = FF

    return voxel

def getStrutlistFromJson(mtype):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    with open(cur_dir + '/line3_valid.json', encoding='utf-8') as f1:
        all_lattices = json.load(f1)
        for lattice in all_lattices:
            if(lattice.get("id") == mtype + 1):
                strut_1_idx = lattice.get("one")
                strut_2_idx = lattice.get("two")
                strut_3_idx = lattice.get("three")
    with open(cur_dir + '/all_lines.json', encoding='utf-8') as f2:
        all_lines = json.load(f2)
        now_strutlist = []
        for line in all_lines:
            if(line.get("id") == strut_1_idx):
               now_strutlist.append((line.get("start")-1,line.get("end")-1))
            if(line.get("id") == strut_2_idx):
               now_strutlist.append((line.get("start")-1,line.get("end")-1))
            if(line.get("id") == strut_3_idx):
               now_strutlist.append((line.get("start")-1,line.get("end")-1))
    strutlist = np.array(now_strutlist, dtype=np.int32)
    return strutlist

def showPara(mtype):
    strutlist = getStrutlistFromJson(mtype)+1

    keyPoints = [1, 2, 3, 5, 6, 9, 11]

    in_parameter_names = ['R1','R2','R3']
    out_parameter_names = []
    node_unique = np.unique(strutlist)
    for node in node_unique:
        if 4 < node < 11:
            if node in keyPoints:
                out_parameter_names.append('E' + str(node))
            else:
                in_parameter_names.append('E' + str(node))
            # print('E' + str(node))
        if 10 < node < 15:
            if node in keyPoints:
                out_parameter_names += ['F' + str(node) + '_1', 'F' + str(node) + '_2']
            else:
                in_parameter_names += ['F' + str(node) + '_1', 'F' + str(node) + '_2']
            # print('F' + str(node) + '_1', 'F' + str(node) + '_2')
        if node == 15:
            # print('T1', 'T2', 'T3')
            in_parameter_names += ['T1', 'T2', 'T3']
    return in_parameter_names, out_parameter_names


def setPara(mtype, in_parameter_values, out_parameter_values):
    ### 参数范围0<E{num}<1, 0<F{num}_1+F{num}_2 < 1
    # R表示半径
    # 0<E{num}<1
    # 0<F{num}_1+F{num}_2 < 1
    # 0<T1+T2+T3<1
    # 0<所有参数值<1
    init_parameters = {
        'R1':0.036, 'R2':0.036, 'R3':0.036,
        'E5':0.1,'E6':0.1,'E7':0.1,'E8':0.1,'E9':0.1,'E10':0.1,
        'F11_1':0.2,'F12_1':0.2,'F13_1':0.2,'F14_1':0.2,
        'F11_2':0.2,'F12_2':0.2,'F13_2':0.2,'F14_2':0.2,
        'T1':0.1,'T2':0.1,'T3':0.1
    }

    in_parameter_names, out_parameter_names = showPara(mtype)
    for value, key in zip(in_parameter_values, in_parameter_names):
        init_parameters[key] = value
    for value, key in zip(out_parameter_values, out_parameter_names):
        init_parameters[key] = value
    return init_parameters

def main_f(mtype,radius,offset,sample):
    in_parameter_names, out_parameter_names = showPara(mtype)
    print('in_parameters', in_parameter_names)
    print('out_parameters', out_parameter_names)

    default = offset
    in_parameter_values = [default]*len(in_parameter_names)
    in_parameter_values[0] = radius
    in_parameter_values[1] = radius
    in_parameter_values[2] = radius
    out_parameter_values = [default]*len(out_parameter_names)
    parameters = setPara(mtype, in_parameter_values, out_parameter_values)

    voxel = p2voxel(mtype, parameters, sample)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    np.save(cur_dir + '/mircostruct', voxel)



