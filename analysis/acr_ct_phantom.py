"""ACR CT Phantom — Modules 1-4 faithful port."""
DISPLAY_NAME   = "ACR CT Phantom"
DESCRIPTION    = "Modules 1–4: CT number accuracy, CNR, uniformity, MTF"
PARAMETERS     = {}
NEEDS_DATASETS = True

import os, tempfile, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border


def _pix(ds):
    s = float(getattr(ds,"RescaleSlope",1.0))
    i = float(getattr(ds,"RescaleIntercept",0.0))
    return ds.pixel_array.astype(np.float32)*s + i

def _get_dfov(ds):
    for tag in [(0x0018,0x1100),(0x0018,0x0090)]:
        try:
            v = ds[tag].value
            f = float(v[0] if hasattr(v,'__len__') else v)
            if f > 50: return f
        except: pass
    try:
        ps = ds[0x0028,0x0030].value
        return float(ps[1]) * int(ds[0x0028,0x0011].value)
    except: return 250.0

def _nbb(img): return float(np.max(img[:50,:]))
def _sbb(img): return float(np.max(img[-50:,:]))
def _ebb(img): return float(np.max(img[:,-50:]))
def _wbb(img): return float(np.max(img[:,:50]))
def _nbb_loc(img):
    r,c = np.unravel_index(img[:50,:].argmax(), img[:50,:].shape); return [int(r),int(c)]
def _sbb_loc(img):
    roi=img[-50:,:]; r,c=np.unravel_index(roi.argmax(),roi.shape); return [int(r)+img.shape[0]-50,int(c)]
def _ebb_loc(img):
    roi=img[:,-50:]; r,c=np.unravel_index(roi.argmax(),roi.shape); return [int(r),int(c)+img.shape[1]-50]
def _wbb_loc(img):
    r,c=np.unravel_index(img[:,:50].argmax(),img[:,:50].shape); return [int(r),int(c)]

def _ramp_val(img,DFOV,dx,dy,north=True):
    rows,cols=img.shape
    nx=int(np.clip(round((DFOV/2)/dx),0,cols-1))
    ny_raw=(((DFOV-200)/2)+52.5)/dy if north else (DFOV-(((DFOV-200)/2)+52.5))/dy
    ny=int(np.clip(round(ny_raw),0,rows-1))
    d=max(1,int(round(10/dx)))
    roi=img[max(0,ny-d):min(rows,ny+d),max(0,nx-d):min(cols,nx+d)]
    return float(np.max(roi)) if roi.size>0 else 0.0

def _circ_roi(img,cx,cy,r_mm,dx):
    rows,cols=img.shape
    r_px=max(1,int(round(r_mm/dx)))
    cx_i=int(np.clip(round(cx),r_px,cols-r_px-1))
    cy_i=int(np.clip(round(cy),r_px,rows-r_px-1))
    ys,xs=np.ogrid[:rows,:cols]
    mask=((xs-cx_i)**2+(ys-cy_i)**2)<=r_px**2
    pts=img[mask]; return pts if len(pts)>0 else np.array([0.0],dtype=np.float32)

def _phantom_ct(img):
    north=_nbb_loc(img); south=_sbb_loc(img); east=_ebb_loc(img); west=_wbb_loc(img)
    cns=[int(round((south[i]+north[i])/2)) for i in range(2)]
    cwe=[int(round((east[i]+west[i])/2)) for i in range(2)]
    center=[int(round((cns[i]+cwe[i])/2)) for i in range(2)]
    dNS=south[1]-north[1]; dWE=east[1]-west[1]
    tNS=float(abs(np.arctan((south[0]-north[0])/dNS))) if dNS!=0 else 0.0
    tWE=float(np.arctan((east[0]-west[0])/dWE)) if dWE!=0 else 0.0
    theta=((tNS-np.pi/2)+tWE)/2
    return center,theta,north,south,east,west

def _largest_centroid(mask):
    lbl=label(mask.astype(np.uint8)); rps=regionprops(lbl)
    if not rps: raise ValueError("No region found")
    best=max(rps,key=lambda r:r.area)
    return int(round(best.centroid[0])),int(round(best.centroid[1]))

def _find_mod_slices(sl1,sl4,dZ,n,zlo,zhi):
    direction=1 if sl4>sl1 else -1
    idxs=[]; z=20.0; i=0
    while z<140.0:
        if zlo<z<zhi:
            c=sl1+direction*i
            if 0<=c<n: idxs.append(c)
        z+=dZ; i+=1
    return idxs if idxs else [sl1]

def _make_fig(img,title,vmin,vmax,rois):
    fig,ax=plt.subplots(figsize=(5,5),facecolor="white")
    ax.set_facecolor("white"); ax.imshow(img,cmap="gray",vmin=vmin,vmax=vmax)
    ax.set_title(title,fontsize=10); ax.axis("off")
    for cy,cx,rpx,lbl2 in rois:
        c=plt.Circle((cx,cy),rpx,color="yellow",fill=False,lw=1.5)
        ax.add_patch(c); ax.text(cx+rpx+3,cy,lbl2,color="yellow",fontsize=7,va="center",clip_on=True)
    plt.tight_layout(); return fig


def run(images_sorted: list, metadata: dict, progress_cb=None) -> dict:
    import pydicom
    if not images_sorted: raise ValueError("No images provided.")
    if not isinstance(images_sorted[0], pydicom.Dataset):
        raise ValueError("ACR CT Phantom requires raw DICOM datasets.")
    datasets=images_sorted; ds0=datasets[0]
    DFOV=_get_dfov(ds0); dZ=float(getattr(ds0,"SliceThickness",5.0))
    dx=float(ds0[0x0028,0x0030].value[1]); dy=float(ds0[0x0028,0x0030].value[0])
    x=int(ds0[0x0028,0x0011].value); y=int(ds0[0x0028,0x0010].value)
    n=len(datasets)
    if progress_cb: progress_cb(5)

    t1=[]; t4=[]
    for ds in datasets:
        img=_pix(ds)
        bb=_nbb(img)+_sbb(img)+_ebb(img)+_wbb(img)
        rmp=_ramp_val(img,DFOV,dx,dy,True)+_ramp_val(img,DFOV,dx,dy,False)
        t1.append(bb+rmp); t4.append(bb-rmp)
    sl1=int(np.argmax(t1)); sl4=int(np.argmax(t4))
    if progress_cb: progress_cb(10)

    all_dfs=[]; all_figs=[]; errors={}

    # Module 1
    try:
        img=_pix(datasets[sl1])
        center=_largest_centroid(ndi.binary_fill_holes(img>-20))
        air_c=_largest_centroid(ndi.binary_fill_holes(clear_border(img<-500)))
        tef_c=_largest_centroid(ndi.binary_fill_holes(img>700))
        dCA=np.hypot((center[0]-air_c[0])*dy,(center[1]-air_c[1])*dx)
        dCT=np.hypot((center[0]-tef_c[0])*dy,(center[1]-tef_c[1])*dx)
        dCI=(dCA+dCT)/2
        da=center[1]-air_c[1]; dt=center[1]-tef_c[1]
        t1a=(np.arctan((center[0]-air_c[0])/da)-np.pi/4) if da!=0 else 0.0
        t1b=(np.arctan((center[0]-tef_c[0])/dt)+np.pi/4) if dt!=0 else 0.0
        theta=(t1a+t1b)/2
        water_c=[int(round(center[0]+(dCI/dy)*np.sin(np.pi+theta))),
                 int(round(center[1]+(dCI/dx)*np.cos(np.pi+theta)))]
        poly_c=[int(round(center[0]+(dCI/dy)*np.sin(-np.pi*(3/4)+theta))),
                int(round(center[1]+(dCI/dx)*np.cos(-np.pi*(3/4)+theta)))]
        pmma_c=[int(round(center[0]+(dCI/dy)*np.sin(np.pi*(3/4)+theta))),
                int(round(center[1]+(dCI/dx)*np.cos(np.pi*(3/4)+theta)))]
        r=np.sqrt(200/np.pi); rpx=max(1,int(round(r/dx)))
        def stats(cy,cx2):
            pts=_circ_roi(img,cx2,cy,r,dx)
            return round(float(np.mean(pts)),2),round(float(np.std(pts)),2)
        crit={"Air":(-1000,40),"Water":(0,7),"PMMA":(120,30),"Poly":(-35,15),"Teflon":(990,100)}
        pf=lambda m,nm:"PASS" if abs(m-crit[nm][0])<=crit[nm][1] else "FAIL"
        air=stats(*air_c); tef=stats(*tef_c); wat=stats(*water_c)
        pol=stats(*poly_c); pma=stats(*pmma_c)
        all_dfs.append(pd.DataFrame({"Module":[1],"Slice":[sl1+1],
            "Air":[air[0]],"Air SD":[air[1]],"Air P/F":[pf(air[0],"Air")],
            "Water":[wat[0]],"Water SD":[wat[1]],"Water P/F":[pf(wat[0],"Water")],
            "PMMA":[pma[0]],"PMMA SD":[pma[1]],"PMMA P/F":[pf(pma[0],"PMMA")],
            "Poly":[pol[0]],"Poly SD":[pol[1]],"Poly P/F":[pf(pol[0],"Poly")],
            "Teflon":[tef[0]],"Teflon SD":[tef[1]],"Teflon P/F":[pf(tef[0],"Teflon")]}))
        all_figs.append(_make_fig(img,f"Module 1 — Slice {sl1+1}",-400,400,
            [(air_c[0],air_c[1],rpx,"Air"),(tef_c[0],tef_c[1],rpx,"Teflon"),
             (water_c[0],water_c[1],rpx,"Water"),(poly_c[0],poly_c[1],rpx,"Poly"),
             (pmma_c[0],pmma_c[1],rpx,"PMMA")]))
    except Exception: errors["Module 1"]=traceback.format_exc(); all_dfs.append(pd.DataFrame())
    if progress_cb: progress_cb(30)

    # Module 2
    try:
        img1=_pix(datasets[sl1]); center,theta,north,south,east,west=_phantom_ct(img1)
        mod2=_find_mod_slices(sl1,sl4,dZ,n,45,75)
        r=np.sqrt(100/np.pi); rpx=max(1,int(round(r/dx)))
        ci=[int(round(center[0]+(57.5*np.sin(-(np.pi/2)-theta))/dy)),
            int(round(center[1]+(57.5*np.cos(-(np.pi/2)-theta))/dx))]
        C=np.sqrt(25**2+52.5**2); phi=np.arccos(52.5/C)
        cb=[int(round(center[0]+(52.5*np.sin(-(np.pi/2)-theta-phi))/dy)),
            int(round(center[1]+(52.5*np.cos(-(np.pi/2)-theta-phi))/dx))]
        CNRs=[]
        for idx in mod2:
            im=_pix(datasets[idx])
            ip=_circ_roi(im,ci[1],ci[0],r,dx); bp=_circ_roi(im,cb[1],cb[0],r,dx)
            bg_s=float(np.std(bp))
            CNRs.append(abs(float(np.mean(ip))-float(np.mean(bp)))/bg_s if bg_s>0 else 0.0)
        best=int(np.argmax(CNRs)); bi=mod2[best]; im=_pix(datasets[bi])
        ip=_circ_roi(im,ci[1],ci[0],r,dx); bp=_circ_roi(im,cb[1],cb[0],r,dx)
        all_dfs.append(pd.DataFrame({"Module":[2],"Best Slice":[bi+1],
            "Insert Mean":[round(float(np.mean(ip)),2)],"Insert SD":[round(float(np.std(ip)),2)],
            "BG Mean":[round(float(np.mean(bp)),2)],"BG SD":[round(float(np.std(bp)),2)],
            "Best CNR":[round(CNRs[best],2)],"CNR≥1.0":["PASS" if CNRs[best]>=1.0 else "FAIL"]}))
        all_figs.append(_make_fig(im,f"Module 2 — Slice {bi+1}  CNR={CNRs[best]:.2f}",0,200,
            [(ci[0],ci[1],rpx,"Insert"),(cb[0],cb[1],rpx,"BG")]))
    except Exception: errors["Module 2"]=traceback.format_exc(); all_dfs.append(pd.DataFrame())
    if progress_cb: progress_cb(55)

    # Module 3
    try:
        img1=_pix(datasets[sl1]); center,theta,north,south,east,west=_phantom_ct(img1)
        mod3=_find_mod_slices(sl1,sl4,dZ,n,85,115)
        cs=mod3[int(np.argmax([np.max(_pix(datasets[i])) for i in mod3]))]
        im=_pix(datasets[cs])
        rc=np.sqrt(400/np.pi); d=np.sqrt(400*4/np.pi); rpx=max(1,int(round(rc/dx)))
        rows2,cols2=im.shape
        def clamp(c): return [int(np.clip(c[0],rpx,rows2-rpx-1)),int(np.clip(c[1],rpx,cols2-rpx-1))]
        N_c=clamp([int(round(north[0]+(1.5*d)/dy)),north[1]])
        S_c=clamp([int(round(south[0]-(1.5*d)/dy)),south[1]])
        E_c=clamp([east[0],int(round(east[1]-(1.5*d)/dx))])
        W_c=clamp([west[0],int(round(west[1]+(1.5*d)/dx))])
        def st(cy,cx2,r2):
            pts=_circ_roi(im,cx2,cy,r2,dx); return round(float(np.mean(pts)),2),round(float(np.std(pts)),2)
        ctr=st(center[0],center[1],rc); nrt=st(N_c[0],N_c[1],rc); sth=st(S_c[0],S_c[1],rc)
        est=st(E_c[0],E_c[1],rc); wst=st(W_c[0],W_c[1],rc)
        diffs=[nrt[0]-ctr[0],sth[0]-ctr[0],est[0]-ctr[0],wst[0]-ctr[0]]
        mnu=round(float(np.max([abs(min(diffs)),abs(max(diffs))])),2)
        all_dfs.append(pd.DataFrame({"Module":[3],"Slice":[cs+1],
            "Center":[ctr[0]],"North":[nrt[0]],"South":[sth[0]],"East":[est[0]],"West":[wst[0]],
            "Max ΔHU":[mnu],"ΔHU≤5":["PASS" if mnu<=5 else "FAIL"]}))
        all_figs.append(_make_fig(im,f"Module 3 — Slice {cs+1}  ΔHU={mnu}",-100,100,
            [(center[0],center[1],rpx,"C"),(N_c[0],N_c[1],rpx,"N"),
             (S_c[0],S_c[1],rpx,"S"),(E_c[0],E_c[1],rpx,"E"),(W_c[0],W_c[1],rpx,"W")]))
    except Exception: errors["Module 3"]=traceback.format_exc(); all_dfs.append(pd.DataFrame())
    if progress_cb: progress_cb(75)

    # Module 4
    try:
        img=_pix(datasets[sl4]); center,theta,north,south,east,west=_phantom_ct(img)
        r=np.sqrt(100/np.pi); rpx=max(1,int(round(r/dx)))
        am={4:-3*np.pi/4,5:-np.pi,6:-5*np.pi/4,7:-3*np.pi/2,
            8:-7*np.pi/4,9:0.0,10:-np.pi/4,12:-np.pi/2}
        def lp_pos(b):
            return [int(round(center[0]+70.0*np.sin(b-theta)/dy)),
                    int(round(center[1]+70.0*np.cos(b+theta)/dx))]
        lpc={lp:lp_pos(am[lp]) for lp in am}
        def rstat(cy,cx2):
            pts=_circ_roi(img,cx2,cy,r,dx); return float(np.mean(pts)),float(np.std(pts))
        cm,cs2=rstat(center[0],center[1])
        bar4=_circ_roi(img,lpc[4][1],lpc[4][0],r,dx)
        bo=bar4[bar4>1000]; M0=(float(np.mean(bo))-cm)/2 if len(bo)>0 else 1.0
        if M0<=0: M0=1.0
        freqs=[0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2]; lps=[4,5,6,7,8,9,10,12]
        MTF=[]
        for lp in lps:
            cy2,cx2=lpc[lp]; _,std=rstat(cy2,cx2)
            M=np.sqrt(max(std**2-cs2**2,0.0))
            MTF.append(round(float((np.pi*np.sqrt(2)/4)*(M/M0)),3))
        passing=[freqs[i] for i,m in enumerate(MTF) if m>0.05]
        highest=f"{int(max(passing)*10)} lp/cm" if passing else "< 4 lp/cm"
        row={"Module":[4],"Slice":[sl4+1],"Highest resolvable":[highest]}
        for i2,f in enumerate(freqs): row[f"{int(f*10)} lp/cm"]=[MTF[i2]]
        all_dfs.append(pd.DataFrame(row))
        rois2=[(lpc[lp][0],lpc[lp][1],rpx,str(lp)) for lp in lps]
        all_figs.append(_make_fig(img,f"Module 4 — Slice {sl4+1}",1000,1200,rois2))
        fig2,ax2=plt.subplots(figsize=(6,4),facecolor="white"); ax2.set_facecolor("white")
        ax2.scatter(freqs,MTF,color="steelblue",zorder=3)
        ax2.plot(freqs,MTF,"--",color="steelblue",alpha=.6)
        ax2.axhline(.05,ls="--",lw=.9,color="red",label="MTF=0.05")
        for f,m in zip(freqs,MTF): ax2.text(f-.015,m+.015,f"{m:.2f}",fontsize=8)
        ax2.set_xticks(freqs); ax2.set_xticklabels([str(int(f*10)) for f in freqs])
        ax2.set_xlabel("lp/cm"); ax2.set_ylabel("MTF"); ax2.set_title("Module 4 MTF")
        ax2.grid(True,alpha=.3); ax2.legend(); plt.tight_layout()
        all_figs.append(fig2)
    except Exception: errors["Module 4"]=traceback.format_exc(); all_dfs.append(pd.DataFrame())
    if progress_cb: progress_cb(100)

    return {"dataframe": all_dfs[0] if all_dfs else pd.DataFrame(),
            "df_module1":all_dfs[0] if len(all_dfs)>0 else pd.DataFrame(),
            "df_module2":all_dfs[1] if len(all_dfs)>1 else pd.DataFrame(),
            "df_module3":all_dfs[2] if len(all_dfs)>2 else pd.DataFrame(),
            "df_module4":all_dfs[3] if len(all_dfs)>3 else pd.DataFrame(),
            "figures":all_figs, "slices_used":[sl1+1,sl4+1], "roi_coords":[], "errors":errors}
