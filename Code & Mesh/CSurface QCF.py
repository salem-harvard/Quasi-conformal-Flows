# -*- coding: utf-8 -*-
"""
Created on Thursday Feb 23 2023

Author: Salem Mosleh
"""

import jax.numpy as np
import numpy.random as onpr
from jax.numpy import cos, sin

import jax.numpy.linalg as la
import jax.random as npr
key = npr.PRNGKey(0)

import numpy as onp

import jax
jax.config.update("jax_enable_x64", True)

from jax import vmap, jit

from jaxopt import ProjectedGradient
from jaxopt.projection import projection_affine_set
import jaxopt




#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
from pathlib import WindowsPath
from pathlib import Path
import os
from os import listdir
from os.path import isfile, join

cwd = Path(os.getcwd())

resultsDir = cwd  / "Run Results" 

meshDir = cwd / "Mesh Data"
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------




runNames = ["sphere_ellipsoid", "david_to_sphere"]
paramterTypes = ["conformal", "elastic", "uniform"]

spatialDim = 3; numTimeSteps = 30



#===========================================================================================================
#   Run optimization for all time steps
#===========================================================================================================
def allTimeSteps(runIndx=0, paramIndex=0, startIndx=0):
    '''
    allTimeSteps(runIndx=0, a=10, b=0,  dGrad=1, sGrad=1, regC=1000)
    
    calculates the optimal registration by minizing the given cost. For each time step
    which is indexed by runIndx. 
    
    a: dilation cost coefficient. which also may include growth costs 
    
    b: shear cost coefficient.
    
    dGrad: dilation gradient cost coefficient.
    
    sGrad: shear gradient cost coefficient.
    
    regC: enforces the constraint that fixed the normal displacment. Only tangent displacements
    are free to move, corresponding to different registrations (diffeomorphisms).
    
    '''
    
    runName = runNames[runIndx]
    paramterType = paramterTypes[paramIndex]
    if paramterType=="conformal":
        a=0; b=1; c=0;d=0;e=0; a3=0; regC=100000;
    elif paramterType=="conformal-forced":
        a=0; b=10; c=0;d=0;e=0; a3=0; regC=100;
    elif paramterType == "elastic":
        a=1; b=1; c=0;d=0;e=0; a3=1; regC=100000; 
    elif paramterType=="uniform":
        a=0; b=0; c=0.1;d=0.1;e=0.0; a3=0; regC=100000; 
    else: 
        raise TypeError("no paramter choise given")
    
    

    
    
    if not os.path.exists(resultsDir / runName):
        os.makedirs(resultsDir / runName)
    
    
    allMaxDeformation = 0
    
    print("working on data for: " + runName)
    
    for stepNum in range(startIndx,numTimeSteps):
        
        mesh_name =   paramterType + "_time" + str(stepNum)
        
        print("working on: " + mesh_name + "_" + runName)
        
        newVerts, initStrains , finalStrains, maxDeformation = stepInTime(stepNum=stepNum, runIndx=runIndx,
                             paramIndex=paramIndex, a=a, b=b, c=c, d=d, e=e,a3=a3, regC=regC)
    
        
        

        
        print("saving results for step number: " + str(stepNum))


        onp.savetxt(resultsDir / runName / ("init_strains_" + mesh_name + ".csv") , initStrains, delimiter=',')   
        onp.savetxt(resultsDir / runName / ("strains_" + mesh_name + ".csv") , finalStrains, delimiter=',')

            
        onp.savetxt(resultsDir / runName / ("new_verts_" + mesh_name + ".csv") , newVerts, delimiter=',')
        
        allMaxDeformation = onp.max([allMaxDeformation, maxDeformation])
  
   
  
    
    onp.savetxt(resultsDir / runName / (paramterType + "_MaxDilations.csv") , [allMaxDeformation], delimiter=',')  

  
    return 
#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#===========================================================================================================


    
#===========================================================================================================
#   Run optimization for a single time step
#===========================================================================================================
def stepInTime(stepNum, runIndx, paramIndex, a, b, c, d, e, a3, regC):
    '''
    '''
    

    
    runName = runNames[runIndx]
    paramterType = paramterTypes[paramIndex]
    
    loadDir = meshDir / runName / ("time" + str(stepNum))    
    
    vertices0 = onp.loadtxt(loadDir / "vertices.csv", delimiter=",", dtype=np.float64)
    
    vertices1 = onp.loadtxt(loadDir / "normal_deformed_intersection.csv", delimiter=",", dtype=np.float64)
    
    faces = onp.loadtxt(loadDir / "faces.csv", delimiter=",", dtype=np.int32)
    numFaces = len(faces)
    
    normalDistances = onp.loadtxt(loadDir / "distance.csv", delimiter=",", dtype=np.float64)

    
    normals = onp.loadtxt(loadDir / "normal.csv", delimiter=",", dtype=np.float64)
    
    gradCostMat = onp.loadtxt(loadDir / "gradCostMat.csv", delimiter=",", dtype=np.float64)
   
    areas = onp.loadtxt(loadDir / "faceAreas.csv", delimiter=",", dtype=np.float64)
    areaMat = np.diag(areas)
    
    
    loadDir0 = meshDir / runName / ("time" + str(0))   
    initialArea = np.sum(onp.loadtxt(loadDir0 / "faceAreas.csv", delimiter=",", dtype=np.float64))
    duration = 30
    
    
    vertexAreas = onp.loadtxt(loadDir / "vertexAreas.csv", delimiter=",", dtype=np.float64)
    
    
    
    #when penalizing variations in time
    if(stepNum==0):
        E = 0
        previousStrains = onp.zeros((numFaces, spatialDim, spatialDim))
    else:
        E = e
        prevStepNum = onp.max([stepNum - 1, 0])
        file_name =   "strains_" + paramterType + "_time" + str(prevStepNum) + ".csv"
        previousStrains = onp.loadtxt(resultsDir / runName / file_name ,  delimiter=',')            
        previousStrains = onp.reshape(previousStrains, (numFaces, spatialDim, spatialDim)) 
    
    
    
    displacements = vertices1 - vertices0
    
     
    
    solver  = jaxopt.LBFGS(fun=cost)
    res = solver.run(displacements, startingVerts=vertices0, initDisps=displacements, 
                      faces=faces, areaMat=areaMat,normals=normals, vertexAreas=vertexAreas,
                       gradCostMat=gradCostMat, prevStrains=previousStrains, 
                       initialArea=initialArea, duration=duration, a=a, b=b, c=c,  d=d, e=E, a3=a3, regC=regC)
     
     
    newDisps, s = res
    
    
    initStrains = duration*getDeformations(vertices0, displacements, faces)[0]
    
    finalStrains = duration*getDeformations(vertices0, newDisps, faces)[0]
    
    newVerts = vertices0 + newDisps
    
    
    maxDeformation = getMaxDilationAndShear(finalStrains)
    
    return newVerts, onp.reshape(initStrains, (numFaces, spatialDim**2)), onp.reshape(finalStrains, (numFaces, spatialDim**2)), maxDeformation
#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#===========================================================================================================





#===========================================================================================================
#   Add the elastic cost for all simplices
#===========================================================================================================
@jit
def cost(disps, startingVerts, initDisps, faces, areaMat,normals, 
                 vertexAreas, gradCostMat, prevStrains, 
                 initialArea, duration, a, b, c, d, e, a3, regC) :
    '''
    '''
    
    strains, bendingStrain = getDeformations(startingVerts, disps, faces) #strains, newNormals = 
    
    cost = 0

    dilationMat = (duration/initialArea) * (a - b/2)*areaMat + duration * (c - d/2)*gradCostMat
    shearMat = (duration/initialArea) * b * areaMat + duration * d * gradCostMat
    
    cost += np.einsum('kii, kl, ljj', strains , dilationMat, strains)
    cost += np.einsum('kij, kl, lij', strains , shearMat, strains)
    
    
    #the bending energy cost
    areas = np.diag(areaMat)
    cost += (duration/initialArea) * a3*(np.einsum('ki,k ,ki -> k', bendingStrain , areas, bendingStrain)).sum()


    #penalize changes over time of the deformations
    strainChange = strains - prevStrains
    changeParam = e*(duration**3)/initialArea
    cost += changeParam*np.einsum('kii, kl, ljj', strainChange , areaMat, strainChange)
    cost += changeParam*np.einsum('kij, kl, lij', strainChange , areaMat, strainChange)
    
    

    
    
    #enforce the normal displacements of the vertices
    #cost += (duration/initialArea**2) * regC * ( vertexAreas*(np.einsum('ij, ij -> i', disps , normals) - normalDistances)**2 ).sum()
    cost += (duration/initialArea**2) * regC * ((
        np.einsum('ij, i,ij -> i', disps - initDisps, vertexAreas, normals)**2).sum())
    
    
    return cost
#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#===========================================================================================================



#===========================================================================================================
# Calculate the shear and dilation given the initial verts and displacements
#===========================================================================================================   
@jit
def getDeformations(vertices, displacements, faces):

    
    v_mapped = vmap(cost_per_simplex2, (0, None, None))
    
        
    return  v_mapped(faces, vertices, displacements) 
#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#=========================================================================================================== 




#===========================================================================================================
#   Calculate the elastic cost per simplex using face index
#===========================================================================================================
@jit
def cost_per_simplex2(face, allVerts, allDisps):

    '''
    This will give the contribution to the cost (or metric) from each simplex
    
    init_positions: 
    List with shape (3,3) of the positions of the vertices of the triangle on 
    the original plane.
    
    displacements: 
    List with shape (3,3) of the changes in positions of the vertices of the triangle on 
    the deformed surface
    
    a: is the coefficient of the area preserving changes. 
    b: is the coefficient of area changes
    '''
    
    init_positions = allVerts[face]
    displacements = allDisps[face]
    
  
    return calculateStrains(init_positions, displacements)
    


#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#===========================================================================================================   


    


#===========================================================================================================
#   Calculate the elastic cost per simplex
#===========================================================================================================
@jit
def calculateStrains(init_positions, displacements):

    '''
    This will give the contribution to the cost (or metric) from each simplex
    
    init_positions: 
    List with shape (3,3) of the positions of the vertices of the triangle on 
    the original plane.
    
    displacements: 
    List with shape (3,3) of the changes in positions of the vertices of the triangle on 
    the deformed surface
    
    a: is the coefficient of the area preserving changes. 
    b: is the coefficient of area changes
    '''
    
    vb1 = init_positions[1] - init_positions[0]
    vb2 = init_positions[2] - init_positions[0]
    
    basis = np.array([vb1, vb2]).transpose()
    dualBasis = la.pinv(basis)
    
    
    deltaV1 = displacements[1] - displacements[0]
    deltaV2 = displacements[2] - displacements[0]
    
    deltaV = np.array([deltaV1, deltaV2]).transpose()
    
 
    
    projection = np.dot(basis, dualBasis)
    
    strain = np.dot(projection, np.dot(deltaV, dualBasis))
    
    oldNormals = np.cross(vb1, vb2);
    oldNormals = oldNormals/la.norm(oldNormals)
    
    
    newNormal = np.cross(vb1 + deltaV1, vb2 + deltaV2);
    newNormal = newNormal/la.norm(newNormal)
    
    return (strain + strain.transpose())*0.5, newNormal - oldNormals
    
#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#===========================================================================================================    



#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#=========================================================================================================== 
def getMaxDilationAndShear(strain):
    
    #finding the maximum dilation
    dilations = np.abs(np.einsum('kii -> k', strain)); maxDilation = onp.max(dilations)
    
    
    shears = np.sqrt(2*np.einsum('kij, kij -> k', strain, strain) - dilations**2)
    maxShear = onp.max(shears)
    
    maxDeformation = onp.max([maxDilation, maxShear])
    
    return maxDeformation
#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#=========================================================================================================== 



    

    
# for i in range(len(runNames)):
#     for j in range(len(paramterTypes)):
#         allTimeSteps(runIndx=i, paramIndex=j)
 

# allTimeSteps(runIndx=1, paramIndex=1, startIndx=19)


# for j in range(len(paramterTypes)):
#     allTimeSteps(runIndx=1, paramIndex=j)



for i in range(len(runNames)):
    allTimeSteps(runIndx=i, paramIndex=2)  
    
# allTimeSteps(runIndx=0, paramIndex=2)    
    
# allTimeSteps(runIndx=0, paramIndex=2)   
    
    
    
    