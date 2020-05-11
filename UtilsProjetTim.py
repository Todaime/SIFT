#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import matplotlib.pyplot as plt
import cv2 as cv
import time
import pylab as LA


# Fonction pour obtenir $G(x,y,\sigma) = \frac{1}{2\sigma^2\pi}e^{-\frac{x^2+y^2}{2\sigma^2}}$ masque de taille nxm à appliquer sur une image de taille nxm.
def Gaussienne(row,col,sigma):
    x,y = np.mgrid[-row//2+1:row//2+1,-col//2+1:col//2+1]
    G = np.exp(-(x**2+y**2)/(2.*sigma**2))/(math.pi*2.0*sigma**2)
    return G/G.sum()


# Fonction pour obtenir $L(x,y,\sigma) = G(x,y,\sigma)*I(x,y)$
def Gradient_Facteur_Echelle(img,sigma) :
    row,col = img.shape[0],img.shape[1]
    G = Gaussienne(row,col,sigma)
    L = cv.filter2D(img,-1,G)
    return L


# Fonction pour obtenir les $L(x,y,\sigma)$ d'un octave.
def Elements_octave(img,sigma_initial,nbElements,k) :
    elements = list()
    sigma = sigma_initial
    elements.append(img)
    premier_element = img
    gaussienne = Gaussienne(premier_element.shape[0],premier_element.shape[1],k)
    for i in range(1,nbElements):
        sigma = sigma*k
        elements.append(cv.filter2D(elements[-1],-1,gaussienne))
              
    return elements


# Fonction pour créer la pyramide d'octave.
def Octaves (image,sigma_initial,s,k,nb_octave) :
    octaves = list()
    sigma = sigma_initial
    img = image.copy()
    img = Gradient_Facteur_Echelle(img,sigma) # Pre lissage sigma init  = 1.6
    for i in range(0,nb_octave):
        octaves.append(Elements_octave(img,sigma,s+3,k))
        img = octaves[-1][2]
        img = img[::2,::2]
    
    return octaves


# Fonction créant la pyramide de difference de gaussienne par octave
def Differences_de_Gaussiennes (image,octaves) :
    octaves_differences = list()
    for gaussiennes_de_octave in octaves :
        differences_de_octave = list()
        for j in range(1,len(gaussiennes_de_octave)):
            differences_de_octave.append(gaussiennes_de_octave[j]-gaussiennes_de_octave[j-1])
        DOGs_octave = np.concatenate([DOGs[:,:,np.newaxis]for DOGs in differences_de_octave], axis=2)
        octaves_differences.append(DOGs_octave)
    return  octaves_differences


# Fonction pour obtenir les points de la forme $(x,y,\sigma)$ maximums par rapport à leur 27 voisins.
# Return : Liste de points candidats de l'octave, de la forme (x,y,k) (x,y) coordonnées du pixel, k indice de la troisième dimension de la matrice D(x,y,sigma) du point.
def Point_candidats_par_octave(diffs_Gauss_octave):
    pointsCandidats = list()
    indicePixel = 13
    for x in range(1,diffs_Gauss_octave.shape[0]-1):
        for y in range(1,diffs_Gauss_octave.shape[1]-1):
            for k in range(1,diffs_Gauss_octave.shape[2]-1):
                voisins = diffs_Gauss_octave[x-1:x+2,y-1:y+2,k-1:k+2]
                if ((indicePixel==np.argmax(voisins) and np.sum(voisins==np.max(voisins))==1 ) or indicePixel==np.argmin(voisins) and np.sum(voisins==np.min(voisins))==1) :
                    pointsCandidats.append((x,y,k))
    return pointsCandidats


# Fonction pour calculer l'offset, la jacobienne et la matrice Hessienne pour tout les points candidats.
def offset_point_candidat(x,y,idInOctave,DOGs_octave) :
        
        DOG_prev = DOGs_octave[:,:,idInOctave-1]
        DOG = DOGs_octave[:,:,idInOctave]
        DOG_next = DOGs_octave[:,:,idInOctave+1]
        
        dx = (DOG[x+1,y]-DOG[x-1,y])/2.0
        dy = (DOG[x,y+1]-DOG[x,y-1])/2.0
        ds = (DOG_next[x,y]-DOG_prev[x,y])/2.0
                
        dyy = DOG[x,y+1]-2.*DOG[x,y]+DOG[x,y-1]
        dxy = ((DOG[x+1,y+1]-DOG[x+1,y-1])-(DOG[x-1,y+1]-DOG[x-1,y-1]))/4.0
        dxs = ((DOG_next[x+1,y]-DOG_next[x-1,y])-(DOG_prev[x+1,y]-DOG_prev[x-1,y]))/4.0
        dxx = DOG[x+1,y]-2.*DOG[x,y]+DOG[x-1,y]
        dys = ((DOG_next[x,y+1]-DOG_next[x,y-1])-(DOG_prev[x,y+1]-DOG_prev[x,y-1]))/4.0
        dss = DOG_next[x,y]-2.*DOG[x,y]+DOG_prev[x,y]
        
        J = np.array([dx,dy,ds])
        H = np.array([[dxx,dxy,dxs],[dxy,dyy,dys],[dxs,dys,dss]])
        offset = -(np.linalg.inv(H)).dot(J)
        
        return offset, J, H[0:2,0:2]


# Cette fonction permet d'obtenir les points clés pour un octave.
def Points_Cles_Octave(DOGS_octave, R_th, seuil,sigma_initial,IndiceOctave):
  points_candidats = Point_candidats_par_octave(DOGS_octave)
  points_cles = []
  points_candidat_selectionne = []
  for point_candidat in points_candidats:
    x, y ,k = point_candidat[0], point_candidat[1],point_candidat[2]
    offset, J, H = offset_point_candidat(x, y, k,DOGS_octave)
    contrast = DOGS_octave[x,y,k] + .5*J.dot(offset) 
    if abs(contrast) > seuil:
        alphaBeta =  H[0,0]*H[1,1]-H[0,1]**2
        alphaPlusBeta = H[0,0] + H[1,1]
        R = alphaPlusBeta**2 / alphaBeta 
        if R < R_th:
            point_cle = np.array([x, y,(sigma_initial*(2**IndiceOctave))*2**(k/2)]) + offset
            points_cles.append(point_cle)
            points_candidat_selectionne.append(point_candidat)
  return np.array(points_cles),points_candidat_selectionne


# Cette fonction permet d'obtenir les points candidats pour tous les octaves (Ont passés le test d'extremum et d'élimination des effets de contour).
# Return : - Liste de tuple, 1 tuple/octave, 1 tuple = ensembles des points clés de l'octave, 1 point clé = (x,y,k), k sigma
#          - Liste de tuple, 1 tuple/octave, 1 tuple = ensembles des points candidats (x,y,k)
def obtention_Points_cles(DOGS,seuil,R_th,sig_Init):
    pointCles = list()
    points_candidat_selectionnes = list()
    for i in range(0,len(DOGS)) :
        pointCles_Octave, points_candidat_selectionnes_octave = Points_Cles_Octave(DOGS[i], R_th, seuil,sig_Init,i)
        pointCles.append(pointCles_Octave)
        points_candidat_selectionnes.append(points_candidat_selectionnes_octave)
    return pointCles,points_candidat_selectionnes

# Function faisant la conversion d'un point de coordonnée cartesien en coordonnée polaire
def cart_to_polar_grad(dx, dy): 
  m = np.sqrt(dx**2 + dy**2) 
  theta = (np.arctan2(dx, dy)+np.pi) * 180/np.pi 
  return m, theta 

# Function retournant le gradient en coorodonnée polaire d'un point associé à une fonction L
def get_grad(L, x, y): 
  dx = L[min(L.shape[0]-1, x+1),y] - L[max(0, x-1),y] 
  dy = L[x,min(L.shape[1]-1, y+1)] - L[x,max(0, y-1)] 
  return cart_to_polar_grad(dx, dy)

# Cette fonction associe la parabole aux trois pics principaux de l'histogramme
def parabole(hist, num_case, largeur_case): 
  valeur_centrale = num_case*largeur_case + largeur_case/2. 
  if num_case == len(hist)-1: valeur_droite = 360 + largeur_case/2. 
  else: valeur_droite = (num_case+1)*largeur_case +largeur_case/2. 
  if num_case == 0: valeur_gauche = -largeur_case/2. 
  else: valeur_gauche = (num_case-1)*largeur_case + largeur_case/2. 
  A = np.array([ 
    [valeur_centrale**2, valeur_centrale, 1], 
    [valeur_droite**2, valeur_droite, 1], 
    [valeur_gauche**2, valeur_gauche, 1]]) 
  b = np.array([ 
    hist[num_case], 
    hist[(num_case+1)%len(hist)], 
    hist[(num_case-1)%len(hist)]]) 
  x = np.linalg.lstsq(A, b, rcond=None)[0] 
  if x[0] == 0: x[0] = 1e-6 
  return -x[1]/(2*x[0])

# Cette fonction convertis l'angle donné en une case d'histogramme
def quantize_orientation(theta, nombre_case): 
  largeur_case = 360//nombre_case 
  return int(np.floor(theta)//largeur_case)

# Cette fonction a pour but d'associer la valeur k modifiée par offset à une valeur dans sa game d'octave
def trouverValeurProche(valeur,valeurs):
    res = valeurs[0]
    for i in valeurs:
        if (abs(valeur-i)<=abs(valeur-res)):
            res = i
    return res

# Cette fonction ajoute l'orientation aux points clés
def obtenir_points_cles_orientes(points_cles,octaves,nbOctave,sigma_initial):
    points_cles_orientes = []
    for i in range (0,nbOctave):
        points_cles_orientes.append(orientation(points_cles[i], octaves[i],i,sigma_initial))
    return points_cles_orientes

def orientation(points_cles, octave, indice_octave,sigma_initial,nombre_case=36):
    points_cles_orientes = []
    largeur_case = 10
    sigma_seuil_octave = sigma_initial*2**(indice_octave)
    row, col = octave[0].shape
    
    valeurs_sigma_de_octave = []
    for i in range (0,5):
        valeurs_sigma_de_octave.append(sigma_seuil_octave*2**(i/2))
        
    masques = []
    for i in range (0,5):
        masques.append(Gaussienne(row,col,valeurs_sigma_de_octave[i]*1.5))
        
    for point_cle in points_cles :
        # On obtient les valeurs arrondie 
        x_ptn = int(round(point_cle[0]))
        y_ptn =  int(round(point_cle[1]))
        k = trouverValeurProche(point_cle[2],valeurs_sigma_de_octave)
        indice = int(math.log2(k/sigma_seuil_octave)*2)
        
        if (x_ptn>row or y_ptn>col or x_ptn<0 or y_ptn<0) : continue
            
        L = octave[indice]
        
        sigma = k*1.5 
        
        w = int(2*np.ceil(sigma)+1) 
        
        hist = np.zeros(nombre_case, dtype=np.float32)
        for oy in range(-w, w+1): 
          for ox in range(-w, w+1): 
            
            x, y = x_ptn+ox, y_ptn+oy 
            
            if x < 0 or x > L.shape[0]-1: continue 
            elif y < 0 or y > L.shape[1]-1: continue 
            elif ox+row//2+w>=row or oy+col//2+w>=col: continue
                
            m, theta = get_grad(L, x, y)
            ponderation = masques[indice][ox+row//2+w, oy+col//2+w] * m 
            case = quantize_orientation(theta, nombre_case) 
            hist[case] += ponderation 
            
        case_max = np.argmax(hist) 
        points_cles_orientes.append([x_ptn, y_ptn, k, parabole(hist, case_max, largeur_case)]) 
        max_val = np.max(hist) 
        for num_case, val in enumerate(hist): 
            if num_case == case_max: continue 
            if (.8*max_val<=val): 
                points_cles_orientes.append([x_ptn, y_ptn, k, parabole(hist, num_case, largeur_case)])
    return np.array(points_cles_orientes)

# Calculer les gradients dans la fenetre
def get_patch_grads(p): 
  r1 = np.zeros_like(p) 
  r1[-1] = p[-1] 
  r1[:-1] = p[1:] 
  r2 = np.zeros_like(p) 
  r2[0] = p[0] 
  r2[1:] = p[:-1] 
  dx = r1-r2 
  r1[:,-1] = p[:,-1] 
  r1[:,:-1] = p[:,1:] 
  r2[:,0] = p[:,0] 
  r2[:,1:] = p[:,:-1] 
  dy = r1-r2 
  return dx, dy

# Cette fonction créer l'histogramme de chaque sous région
def get_histogram_for_subregion(i,m, theta, num_bin, reference_angle, bin_width, subregion_w): 
  hist = np.zeros(num_bin, dtype=np.float32) 
  c = subregion_w/2 - .5
  for mag, angle in zip(m, theta):
    angle = (angle-reference_angle) % 360        
    binno = quantize_orientation(angle, num_bin)        
    vote = mag      
   
    hist_interp_weight = 1 - abs(angle - (binno*bin_width + bin_width/2))/(bin_width/2)        
    vote *= max(hist_interp_weight, 1e-6)         
    gx, gy = np.unravel_index(i, (subregion_w, subregion_w))        
    x_interp_weight = max(1 - abs(gx - c)/c, 1e-6)            
    y_interp_weight = max(1 - abs(gy - c)/c, 1e-6)        
    vote *= x_interp_weight * y_interp_weight
    hist[binno-1] += vote
  hist /= max(1e-6, LA.norm(hist)) 
  hist[hist>0.2] = 0.2 
  hist /= max(1e-6, LA.norm(hist))
  return hist

# Cette fonction produit les descripteurs locaux
def descripteurs_locaux_octave(points_cles_orientes,octave,indice_octave,sigma_initial, num_subregion=4, w = 16,num_bin=8): 
    
    descs = [] 
    bin_width = 360//num_bin
    row,col = octave[0].shape
    sigma_seuil_octave = sigma_initial*2**(indice_octave)
    NbPoint = len(points_cles_orientes)
    compteur = 1
    masque = Gaussienne(row,col,w/6)
    
    for point_cle_oriente in points_cles_orientes: 
        compteur+=1
        x_point, y_point, s = int(point_cle_oriente[0]), int(point_cle_oriente[1]), point_cle_oriente[2] 
        
        if (x_point>row or y_point>col or x_point<0 or y_point<0) :
            print("Ahah")
            
        indice = int(math.log2(s/sigma_seuil_octave)*2)
        
        L = octave[indice]
 
        t, l = max(0, x_point-w//2), max(0, y_point-w//2) 
        b, r = min(row, x_point+w//2+1), min(col, y_point+w//2+1) 
        
        patch = L[t:b, l:r] 
            
        dx, dy = get_patch_grads(patch)
        
        row_dx = dx.shape[0]
        col_dx = dx.shape[1]
        if col_dx%2!=0 :
            if row_dx%2!=0 :
                masque_ptn = masque[(row//2)-row_dx//2:(row//2)+row_dx//2+1,(col//2)-col_dx//2:(col//2)+col_dx//2+1] 
            else :
                masque_ptn = masque[(row//2)-row_dx//2:(row//2)+row_dx//2,(col//2)-col_dx//2:(col//2)+col_dx//2+1] 
        else :
            if row_dx%2!=0:
                masque_ptn = masque[(row//2)-row_dx//2:(row//2)+row_dx//2+1,(col//2)-col_dx//2:(col//2)+col_dx//2] 
            else :
                masque_ptn = masque[(row//2)-row_dx//2:(row//2)+row_dx//2,(col//2)-col_dx//2:(col//2)+col_dx//2] 
        m, theta = cart_to_polar_grad(dx, dy) 
        dx = dx*masque_ptn
        dy = dy*masque_ptn
        subregion_w = w//num_subregion 
        featvec = np.zeros(num_bin * num_subregion**2, dtype=np.float32) 
        for i in range(0, subregion_w): 
          for j in range(0, subregion_w): 
            t, l = i*subregion_w, j*subregion_w 
            b, r = min(L.shape[0], (i+1)*subregion_w), min(L.shape[1], (j+1)*subregion_w) 
            hist = get_histogram_for_subregion(i,m[t:b, l:r].ravel(), theta[t:b, l:r].ravel(), num_bin,point_cle_oriente[3],bin_width,subregion_w) 
            featvec[i*subregion_w*num_bin + j*num_bin:i*subregion_w*num_bin + (j+1)*num_bin] = hist.flatten() 
        featvec /= max(1e-6, LA.norm(featvec))        
        featvec[featvec>0.2] = 0.2        
        featvec /= max(1e-6, LA.norm(featvec))    
        descs.append(featvec)
    return np.array(descs)

def obtenir_descripteurs_locaux(points_cles,octaves,nbOctave,sigma_init,nombre_sous_region = 4,nombre_case = 8):
    descripteurs_locaux = []
    for i in range (0,nbOctave):
        descripteurs_locaux.append(descripteurs_locaux_octave(points_cles[i], octaves[i],i,sigma_init))
    return descripteurs_locaux
