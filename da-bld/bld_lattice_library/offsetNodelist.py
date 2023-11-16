import numpy as np


def get_nodelist(parameters):
    # 创建15x3的数组，填充为0
    nodelist = np.zeros((15, 3))
    # 填充前四行,对应四面体的四个顶点坐标
    nodelist[:4] = np.array([[0.5, 0.0, 0.5], [0.5, 0.0, 1.0], [1.0, 0.0, 1.0], [0.5, 0.5, 0.5]])

    # 所有可能的杆连接关系
    strutlist = np.array([[1, 2], [2, 3], [3, 4], [4, 1], [1, 3], [2, 4]]) - 1

    # 所有可能的面连接关系
    facelist = np.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]) - 1

    pe = np.array([[parameters['E5']],
          [parameters['E6']],
          [parameters['E7']],
          [parameters['E8']],
          [parameters['E9']],
          [parameters['E10']]])

    pf = np.array([[parameters['F11_1'], parameters['F11_2']],
          [parameters['F12_1'], parameters['F12_2']],
          [parameters['F13_1'], parameters['F13_2']],
          [parameters['F14_1'], parameters['F14_2']]])

    pt = np.array([[parameters['T1'],parameters['T2'],parameters['T3']]])

    # 根据比例参数得到各边点的具体坐标
    for i in range(0, 6):
        nodelist[4 + i, :] = pe[i] * nodelist[strutlist[i, 0]] + (1 - pe[i]) * nodelist[strutlist[i, 1]]

    # 根据比例参数得到各面点的具体坐标
    for i in range(0, 4):
        vec1 = nodelist[facelist[i, 1]] - nodelist[facelist[i, 0]]
        vec2 = nodelist[facelist[i, 2]] - nodelist[facelist[i, 0]]
        nodelist[10 + i] = pf[i, 0] * vec1 + pf[i, 1] * vec2 + nodelist[facelist[i, 0]]

    # 三条边的向量
    vec1 = nodelist[1] - nodelist[0]
    vec2 = nodelist[2] - nodelist[0]
    vec3 = nodelist[3] - nodelist[0]

    # 根据比例参数得到体点的具体坐标
    nodelist[14] = pt[0, 0] * vec1 + pt[0, 1] * vec2 + pt[0, 2] * vec3 + nodelist[0]

    return nodelist


def get_init_nodelist():
    nodelist = np.zeros((15, 3))
    nodelist[:4] = np.array([[0.5, 0.0, 0.5], [0.5, 0.0, 1.0], [1.0, 0.0, 1.0], [0.5, 0.5, 0.5]])

    strutlist = np.array([[1, 2], [2, 3], [3, 4], [4, 1], [1, 3], [2, 4]]) - 1

    facelist = np.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]) - 1

    pe = 0.5 * np.ones((6, 1))

    pf = 0.25 * np.ones((4, 2))

    pt = 0.3 * np.ones((1, 3))

    for i in range(0, 6):
        nodelist[4 + i, :] = pe[i] * nodelist[strutlist[i, 0]] + (1 - pe[i]) * nodelist[strutlist[i, 1]]
        # print(pe[i]*nodelist[strutlist[i,0]] + (1-pe[i])*nodelist[strutlist[i,1]])
        # print(nodelist[4+i,:])

    for i in range(0, 4):
        vec1 = nodelist[facelist[i, 1]] - nodelist[facelist[i, 0]]
        vec2 = nodelist[facelist[i, 2]] - nodelist[facelist[i, 0]]
        nodelist[10 + i] = pf[i, 0] * vec1 + pf[i, 1] * vec2 + nodelist[facelist[i, 0]]

    vec1 = nodelist[1] - nodelist[0]
    vec2 = nodelist[2] - nodelist[0]
    vec3 = nodelist[3] - nodelist[0]
    nodelist[14] = pt[0, 0] * vec1 + pt[0, 1] * vec2 + pt[0, 2] * vec3 + nodelist[0]
    return nodelist

# symm48(nodelist,strutlist):根据微结构进行48对称
def symm48(nodelist,strutlist):

    nodelist_yz = nodelist.copy() # deep copy
    nodelist_yz[:,2] = 1-nodelist[:,1]
    nodelist_yz[:, 1] = 1 - nodelist[:, 2]
    strutlist_yz = strutlist + nodelist.shape[0]
    nodelist = np.concatenate([nodelist,nodelist_yz])
    strutlist = np.concatenate([strutlist, strutlist_yz])

    nodelist_xz = nodelist.copy()
    nodelist_xz[:,2] = nodelist[:,0]
    nodelist_xz[:, 0] = nodelist[:, 2]
    strutlist_xz = strutlist + nodelist.shape[0]
    nodelist = np.concatenate([nodelist,nodelist_xz])
    strutlist = np.concatenate([strutlist, strutlist_xz])

    nodelist_xy = nodelist.copy()
    nodelist_xy[:, 1] = 1-nodelist[:, 0]
    nodelist_xy[:, 0] = 1-nodelist[:, 1]
    strutlist_xy = strutlist + nodelist.shape[0]
    nodelist = np.concatenate([nodelist,nodelist_xy])
    strutlist = np.concatenate([strutlist, strutlist_xy])

    nodelist_z = nodelist.copy()
    nodelist_z[:, 2] = 1-nodelist[:, 2]
    strutlist_z = strutlist + nodelist.shape[0]
    nodelist = np.concatenate([nodelist,nodelist_z])
    strutlist = np.concatenate([strutlist, strutlist_z])

    nodelist_y = nodelist.copy()
    nodelist_y[:, 1] = 1-nodelist[:, 1]
    strutlist_y = strutlist + nodelist.shape[0]
    nodelist = np.concatenate([nodelist,nodelist_y])
    strutlist = np.concatenate([strutlist, strutlist_y])

    nodelist_x = nodelist.copy()
    nodelist_x[:, 0] = 1-nodelist[:, 0]
    strutlist_x = strutlist + nodelist.shape[0]
    nodelist = np.concatenate([nodelist,nodelist_x])
    strutlist = np.concatenate([strutlist, strutlist_x])

    return nodelist, strutlist




