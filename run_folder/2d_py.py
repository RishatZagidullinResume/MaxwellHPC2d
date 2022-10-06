from PyQt5 import QtGui
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph.opengl as gl
from collections import OrderedDict
import sys

animation_on = False
def stop_visual():
    global animation_on
    animation_on = False

def start_visual():
    global animation_on
    animation_on = True

def reset_visual():
    global i, animation_on
    i = 0
    animation_on = False
    iterationvalue.setText('T= {}'.format(i))
    colors = np.genfromtxt("./data/colours_0.txt", skip_header=i*sizes[0], 
                           max_rows=sizes[0], usecols=(0,1,2,3))
    for count in range(1, int(sys.argv[3])):
        name = "./data/colours_{0}.txt".format(count)
        colors = np.concatenate((colors, np.genfromtxt(name, skip_header=i*sizes[count], 
                                max_rows=sizes[count], usecols = (0,1,2,3)) ))
    inds = np.unique(global_ids, return_index=True)[1]
    colors = np.array([colors[index] for index in inds])
    m1.setMeshData(vertexes=verts, faces=faces, faceColors=colors, smooth=False,
                   drawEdges=True, edgeColor=(1,1,1,1)) 

## Always start by initializing Qt (only once per application)
app = QtWidgets.QApplication([])

## Define a top-level widget to hold everything
w = QtWidgets.QWidget()
w.setWindowTitle("simulation")
## Create some widgets to be placed inside
start_btn = QtWidgets.QPushButton('start')
start_btn.clicked.connect(start_visual)
stop_btn = QtWidgets.QPushButton('stop')
stop_btn.clicked.connect(stop_visual)
reset_btn = QtWidgets.QPushButton('reset')
reset_btn.clicked.connect(reset_visual)
## Create a grid layout to manage the widgets size and position
layout = QtWidgets.QGridLayout()
w.setLayout(layout)
w.resize(875, 800)
w.show()
## Add widgets to the layout in their proper positions
i = 0
iterationvalue = QtWidgets.QLabel('T= {}'.format(i))
layout.addWidget(iterationvalue, 1, 1)
layout.addWidget(start_btn, 2, 1)
layout.addWidget(stop_btn, 3, 1)
layout.addWidget(reset_btn, 4, 1)

grid = gl.GLViewWidget()
grid.setCameraPosition(elevation=90, azimuth=0) ##, distance=0)

verts = np.loadtxt("./data/vertices.txt")
faces = np.loadtxt("./data/faces_0.txt", dtype=int)
for ii in range(1, int(sys.argv[2])):
    name = './data/faces_{0}.txt'.format(ii)
    faces = np.concatenate((faces, np.loadtxt(name, dtype=int)))
 
sizes = []   
global_ids = np.loadtxt("./data/global_ids_reference_0.txt", dtype=int)
sizes.append(len(global_ids))

for ii in range(1, int(sys.argv[3])):
    name = "./data/global_ids_reference_{0}.txt".format(ii)
    temp = np.loadtxt(name, dtype=int)
    global_ids = np.concatenate((global_ids, temp))
    sizes.append(len(temp))
    print(sizes[ii], global_ids.shape)

colors = np.genfromtxt("./data/colours_0.txt", max_rows=sizes[0], usecols=(0,1,2,3))
for ii in range(1, int(sys.argv[3])):
    name = "./data/colours_{0}.txt".format(ii)
    colors = np.concatenate((colors, np.genfromtxt(name, max_rows=sizes[ii],
                            usecols = (0,1,2,3))))
inds = np.unique(global_ids, return_index=True)[1]
colors = np.array([colors[index] for index in inds])

m1 = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=colors, smooth=False,
                   drawEdges=True, edgeColor=(1,1,1,1))
m1.translate(-1.0, -0.5, 0.0)
m1.setGLOptions('additive')
m1.rotate(90, 0,0,1)
grid.addItem(m1)

gw = pg.GradientEditorItem(orientation='right')

gw.restoreState({'ticks': [(0.0, (0, 0, 255, 255)), 
                           (0.5, (0, 255, 0, 255)),
                           (1.0, (255, 0, 0, 255))],
                 'mode': 'rgb'})

ax = pg.AxisItem('left')
ax.setRange(-1.0, 1.0)
cb = pg.GraphicsLayoutWidget()

cb.addItem(ax)
cb.addItem(gw)
cb.resize(100, 700)
cb.show()
layout.addWidget(cb, 0, 1)
layout.addWidget(grid, 0, 0)
layout.setColumnStretch(0, 1)

def updateData():
    global colors, i, animation_on, verts, faces, total_frames

    QtCore.QTimer.singleShot(100, updateData)

    if (animation_on):
        iterationvalue.setText('T= {}'.format(i))
        
        colors = np.genfromtxt("./data/colours_0.txt", skip_header=i*sizes[0],
                               max_rows=sizes[0], usecols=(0,1,2,3))
        for count in range(1, int(sys.argv[3])):
            name = "./data/colours_{0}.txt".format(count)
            colors = np.concatenate((colors, np.genfromtxt(name, skip_header=i*sizes[count],
                                    max_rows=sizes[count], usecols = (0,1,2,3))))
        inds = np.unique(global_ids, return_index=True)[1]
        colors = np.array([colors[index] for index in inds])
        i = (i+1)%total_frames
        m1.setMeshData(vertexes=verts, faces=faces, faceColors=colors, smooth=False,
                       drawEdges=True, edgeColor=(1,1,1,1))   
        grid.grabFramebuffer().save('./images/fileName_%d.png' % i)

updateData()

if __name__ == '__main__':
    total_frames = int(sys.argv[1])
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()


