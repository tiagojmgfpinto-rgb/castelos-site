"""Classes do chão a partir da cor dos pontos LiDAR (Região Açores, 25 cm). A ortofoto NÃO vai para a maqueta: serve
só para saber o que é cada sítio (asfalto, passeio, relva, mato, terra, branco, tijoleira, betão, rocha). Primeiro as
sombras do voo (sol a 271°, 44°) são compensadas em Lab e os carros tirados, para a classe não mudar à sombra; depois
segmenta-se a partir de traços à mão (dados/chao_tracos.json) e das classes do LiDAR, com as fronteiras encostadas
às arestas da ortofoto, e gravam-se dados/chao_classes.png (uma classe por célula de 25 cm, linha 0 = y 200, como orto.jpg) e dados/chao_tinta.png (tom da relva, seca ou verde, alisado a 3 m)."""
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

# ---------------------------------------------------------------- classes
# Segmentação semeada: os traços à mão de dados/chao_tracos.json (traçados na ortofoto, zona a zona) e sementes do
# LiDAR (classe 11 = estrada, 6 = edifício, 4/5 = vegetação média e alta) crescem por watershed sobre o gradiente da
# ortofoto compensada (Lab) e do SUP, de modo que cada fronteira encosta à aresta real (lancil, muro, beira da relva).
import json, shapely
from PIL import ImageDraw
from skimage import color, filters, segmentation
CL = ['asfalto', 'passeio', 'relva', 'mato', 'terra', 'branco', 'tijoleira', 'betao', 'rocha', 'asfalto2', 'calcada', 'muro']
I = img.astype(np.float32) / 255.0                      # sem sombras nem carros, y para cima
P = 0.25
lab = color.rgb2lab(ndi.gaussian_filter(I, (1.0, 1.0, 0)))
g = sum(filters.sobel(lab[..., i]) * w for i, w in ((0, 1.0), (1, 1.6), (2, 1.6)))
g = g / np.percentile(g, 95) + np.clip(filters.sobel(ndi.gaussian_filter(Z, 0.7)) / P, 0, 3) * 0.8
tr = json.load(open(D + 'chao_tracos.json'))
mk = Image.new('I', (Z.shape[1], Z.shape[0]), 0)
dw = ImageDraw.Draw(mk)
ids = [0]
px = lambda x, y: ((x + 200) / P, (y + 168) / P)
for t in tr:
    if isinstance(t, str):
        continue
    ids.append(CL.index(t['c']))
    pts = [px(*q) for q in t['p']]
    dw.line(pts, fill=len(ids) - 1, width=max(1, int(t.get('w', 0.6) / P)))
mk = np.asarray(mk).astype(np.int32)
L = np.load(D + 'lidar.npz')
lc = np.repeat(np.repeat(L['classe'], 2, 0), 2, 1)[:Z.shape[0], :Z.shape[1]]
er = lambda m, n: ndi.binary_erosion(m, iterations=n)
FT = shapely.Polygon(P_['forte']).buffer(2)
noforte = shapely.contains_xy(FT, XX, YY)
r_, g_, b_ = [ndi.uniform_filter(I[..., i], 5) for i in range(3)]
sm_ = r_ + g_ + b_ + 1e-6
verde = er((g_ / sm_ > 0.37) & (g_ > r_) & ((np.maximum(np.maximum(r_, g_), b_) - np.minimum(np.minimum(r_, g_), b_)) > 0.08)
           & (lc != 11), 3)
alt_ = np.repeat(np.repeat(L['mds'] - L['mdt'], 2, 0), 2, 1)[:Z.shape[0], :Z.shape[1]]   # cm; edifícios: mais de 3 m acima do chão (os muros de suporte são classe 6 com c. 2 m)
estr = er(ndi.binary_closing(lc == 11, iterations=2), 6)   # a classe 11 do voo inclui por vezes o passeio: só o meio é semente
mar = (Z < 1.0) | ((lc == 0) & (Z < 2.2))
mar = np.isin(ndi.label(mar)[0], 1 + np.flatnonzero(ndi.sum(mar, ndi.label(mar)[0], np.arange(1, ndi.label(mar)[1] + 1)) > 400))
# dentro do forte a "estrada" do LiDAR é o caminho de terra batida castanho-arroxeada (ortofoto: R-G c. +10)
auto = (('asfalto', estr & ~noforte), ('terra', estr & noforte), ('muro', er((lc == 6) & (alt_ > 300), 3) & ~noforte),
        ('mato', er(np.isin(lc, (4, 5)), 2)), ('relva', verde), ('rocha', mar & ~noforte))
for nome, m in auto:
    ids.append(CL.index(nome))
    mk[(mk == 0) & m] = len(ids) - 1
k = np.array(ids)[segmentation.watershed(g, mk)].astype(np.uint8)
# fronteiras lisas: cada classe alisada a 0,75 m e argmax (um lancil não serpenteia)
acc = np.stack([ndi.gaussian_filter((k == c).astype(np.float32), 3.0) for c in range(len(CL))])
k = acc.argmax(0).astype(np.uint8)
del acc
# faixas de largura constante (passeios, caminhos) e polígonos fechados (parques), pintados por cima
for t in tr:
    if isinstance(t, str) or ('faixa' not in t and not t.get('poligono')):
        continue
    if t.get('poligono'):
        fx = shapely.Polygon(t['p']).buffer(0)
    else:
        fx = shapely.LineString(t['p']).buffer(t['faixa'] / 2, cap_style='flat', join_style='round')
    k[shapely.contains_xy(fx, XX, YY)] = CL.index(t['c'])
# marcas brancas na estrada: píxeis claros e sem cor, em manchas finas (tinta, não carros) dentro do asfalto
asf = np.isin(k, (0, 9))
brc = (I.mean(2) > 0.55) & ((I.max(2) - I.min(2)) < 0.14) & ndi.binary_erosion(asf, iterations=2)
lab_, n_ = ndi.label(brc)
idx = np.arange(1, n_ + 1)
fino = (ndi.maximum(ndi.distance_transform_edt(brc), lab_, idx) <= 1.6) & (ndi.sum(brc, lab_, idx) >= 4)
# cada marca fica um retângulo nítido (passadeiras, linhas de paragem e de estacionamento); manchas redondas
# (restos de carros, reflexos) saem
for i_ in idx[fino]:
    rr, cc = np.nonzero(lab_ == i_)
    if len(rr) < 6:
        continue
    pts = np.c_[XX[rr, cc], YY[rr, cc]]
    c0 = pts.mean(0)
    w_, v_ = np.linalg.eigh(np.cov((pts - c0).T) + np.eye(2) * 1e-4)
    ax = v_[:, 1]
    u = (pts - c0) @ ax
    t_ = (pts - c0) @ v_[:, 0]
    comp_, larg_ = np.ptp(u) + 0.25, max(np.ptp(t_) + 0.25, 0.15)
    if comp_ < 1.0 or comp_ / larg_ < 2.5:
        continue
    um, tm = (u.max() + u.min()) / 2, (t_.max() + t_.min()) / 2
    cc_ = c0 + ax * um + v_[:, 0] * tm
    ret = shapely.Polygon([cc_ + ax * a * comp_ / 2 + v_[:, 0] * b * larg_ / 2 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    r0_, c0_ = rr.min() - 6, cc.min() - 6
    sl = (slice(max(r0_, 0), rr.max() + 7), slice(max(c0_, 0), cc.max() + 7))
    m_ = shapely.contains_xy(ret, XX[sl], YY[sl]) & asf[sl]
    k[sl][m_] = 5
Image.fromarray(k[::-1]).save(D + 'chao_classes.png')
# tom da relva: seca (amarelada) ou verde, pela cor da ortofoto alisada a 3 m, só nas células de relva
rv = (k == 2) | (k == 3)
w_ = ndi.gaussian_filter(rv.astype(np.float32), 12) + 1e-4
t_ = np.stack([ndi.gaussian_filter(np.where(rv, I[..., c], 0), 12) / w_ for c in range(3)], 2)
t_ = np.ascontiguousarray(t_[::4, ::4])
Image.fromarray((np.clip(t_, 0, 1) * 255).astype(np.uint8)[::-1]).save(D + 'chao_tinta.png')
print('classes', dict(zip(CL, np.bincount(k.ravel(), minlength=len(CL)).tolist())))
