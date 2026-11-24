"""Textura do chão a partir da cor dos pontos LiDAR (Região Açores, 25 cm): sombras do voo
(sol a 271°, 44°) compensadas em Lab e carros tirados. Grava dados/chao.jpg."""
import numpy as np, cv2
from PIL import Image
from scipy import ndimage as ndi
import os
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'dados')+'/'
S=np.load(D+'superficie.npz'); Z=ndi.median_filter(S['z']/100.0,3)   # linhas: y = -168 + i*0.25
O=np.asarray(Image.open(D+'orto.jpg').convert('RGB')).astype(np.float32)[::-1]/255.0  # linhas: y = -168 + i*0.25 (aprox.)
lin=O**2.2
AZ,EL=271.0,44.0
def sombra(Zs,p,maxd=45):
    dx,dy=np.sin(np.radians(AZ)),np.cos(np.radians(AZ)); t=np.tan(np.radians(EL))
    sh=np.zeros(Zs.shape,bool)
    for d in np.arange(p,maxd,p):
        sz=ndi.shift(Zs,(-dy*d/p,-dx*d/p),order=0,mode='nearest')
        sh|= sz > Zs + d*t + 0.25
    return sh
sh=sombra(Z,0.25)
lum=lin.mean(2)
# afinar: sombra geométrica só onde está de facto escuro; e escuro colado à sombra geométrica
viz=ndi.binary_dilation(sh,iterations=14)
esc=ndi.uniform_filter(lum,3)<0.035
m=(sh&(ndi.uniform_filter(lum,3)<0.08))|(viz&esc)
m=ndi.binary_opening(m,iterations=1); m=ndi.binary_closing(m,iterations=2)
# ganho por canal: anel por fora vs anel por dentro, só chão plano
flat=np.hypot(*np.gradient(Z,0.25))<0.2
fora=ndi.binary_dilation(m,iterations=4)&~ndi.binary_dilation(m,iterations=1)&flat
dentro=ndi.binary_erosion(m,iterations=1)&~ndi.binary_erosion(m,iterations=4)&flat
# ganho local (janela 12 m) para respeitar materiais diferentes
k=48
def media(a,msk): 
    s=ndi.uniform_filter(np.where(msk,a,0),k); n=ndi.uniform_filter(msk.astype(float),k); return s/np.maximum(n,1e-4),n
ganho=np.zeros_like(lin)
for c in range(3):
    mf,nf=media(lin[:,:,c],fora); md,nd=media(lin[:,:,c],dentro)
    g_loc=mf/np.maximum(md,1e-4)
    g_glob=np.median(lin[:,:,c][fora])/np.median(lin[:,:,c][dentro])
    w=np.clip(np.minimum(nf,nd)*20,0,1)
    gl=np.full_like(g_loc,g_glob)
    ganho[:,:,c]=gl
print('ganho global',[np.median(lin[:,:,c][fora])/np.median(lin[:,:,c][dentro]) for c in range(3)])
# sombras: em Lab, L e a,b vêm do chão iluminado em volta, separado em verde (relva) e não verde
srgb=(np.clip(lin,0,1)**(1/2.2)*255).astype(np.uint8)
lab_=cv2.cvtColor(srgb,cv2.COLOR_RGB2LAB).astype(np.float32)
comp=np.clip(lin*ganho,0,1)
labc=cv2.cvtColor((comp**(1/2.2)*255).astype(np.uint8),cv2.COLOR_RGB2LAB).astype(np.float32)
verde_lit=(lab_[:,:,1]<124)&~m      # a* negativo = verde
verde_sh=(labc[:,:,1]<126)&m
K=41  # 10 m
res=lab_.copy()
for cat_lit,cat_sh in ((verde_lit,verde_sh),(~verde_lit&~m,m&~verde_sh)):
    nlit=ndi.uniform_filter(cat_lit.astype(np.float32),K)
    nsh=ndi.uniform_filter(cat_sh.astype(np.float32),K)
    for c in range(3):
        mlit=ndi.uniform_filter(np.where(cat_lit,lab_[:,:,c],0),K)/np.maximum(nlit,1e-4)
        msh=ndi.uniform_filter(np.where(cat_sh,labc[:,:,c],0),K)/np.maximum(nsh,1e-4)
        # recurso: médias de toda a imagem quando não há vizinhos iluminados
        gl=np.median(lab_[:,:,c][cat_lit]); gs=np.median(labc[:,:,c][cat_sh])
        okl=nlit>0.02; oks=nsh>0.02
        mlit=np.where(okl,mlit,gl); msh=np.where(oks,msh,gs)
        v=labc[:,:,c]-msh+mlit if c>0 else (labc[:,:,c]-msh)*0.8+mlit
        res[:,:,c]=np.where(cat_sh,v,res[:,:,c])
a_=ndi.gaussian_filter(m.astype(np.float32),1.0)[:,:,None]
fin=lab_*(1-a_)+res*a_
img=cv2.cvtColor(np.clip(fin,0,255).astype(np.uint8),cv2.COLOR_LAB2RGB)
out=(img.astype(np.float32)/255.0)**2.2
# carros: saliências compactas de 0,8 a 2,6 m em zonas planas e baixas (estradas, parques)
op=ndi.grey_opening(Z,size=(13,13))
alto=(Z-op>0.7)&(Z-op<2.8)
lab,n=ndi.label(alto)
car=np.zeros_like(alto)
for i,sl in enumerate(ndi.find_objects(lab),1):
    r=lab[sl]==i; a_=r.sum()*0.0625
    h,w=r.shape
    if 3<a_<16 and max(h,w)*0.25<7.5: car[sl]|=r
import json, shapely
P_=json.load(open(D+'planta.json'))
F_=shapely.Polygon(P_['forte']).buffer(3)
yy,xx=np.mgrid[0:Z.shape[0],0:Z.shape[1]]
fora_forte=~shapely.contains_xy(F_,-200+xx*0.25,-168+yy*0.25)
o_=ndi.uniform_filter(O,(9,9,1)); cinza=(o_.max(2)-o_.min(2))<0.09
car&=fora_forte
# estaleiro dos barcos: os barcos ficam (o inpaint borrava-os)
XX,YY=-200+xx*0.25,-168+yy*0.25
car&=~((XX<-55)&(YY>-82)&(YY<-12))
car=ndi.binary_dilation(car,iterations=3)
img=(np.clip(out,0,1)**(1/2.2)*255).astype(np.uint8)
img=cv2.inpaint(np.ascontiguousarray(img),car.astype(np.uint8)*255,5,cv2.INPAINT_TELEA)
# recorte da base: x -116..116, y -138..94 (232 m) em 2048 px (11,3 cm/px), linha 0 = y 94
X0, Y0, L, N = -116.0, -138.0, 232.0, 2048
c0, r0 = int((X0 + 200) / 0.25), int((Y0 + 168) / 0.25)
rec = img[r0:r0 + int(L / 0.25), c0:c0 + int(L / 0.25)][::-1]
big = cv2.resize(rec, (N, N), interpolation=cv2.INTER_LANCZOS4)
desf = cv2.GaussianBlur(big, (0, 0), 1.2)
big = cv2.addWeighted(big, 1.35, desf, -0.35, 0)
Image.fromarray(big).save(D + 'chao.jpg', quality=90)
print('chao.jpg', big.shape, 'sombras', int(m.sum()), 'carros', int(car.sum()))
