from __future__ import annotations
from pathlib import Path
from uuid import uuid4
import csv, io, json, os, shutil, time
import pandas as pd
import numpy as np
from .store import ROOT, _db

QUOTA_BYTES = 1024**3
DATA_ROOT = ROOT / 'datasets'; DATA_ROOT.mkdir(parents=True, exist_ok=True)

def ensure_dataset_schema():
    with _db() as con:
        con.executescript('''
        CREATE TABLE IF NOT EXISTS datasets(
          id TEXT PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL,
          filename TEXT NOT NULL, path TEXT NOT NULL, size_bytes INTEGER NOT NULL,
          format TEXT NOT NULL, rows INTEGER, columns INTEGER, created_at REAL NOT NULL,
          parent_id TEXT
        );
        ''')

def usage(user_id:str)->int:
    ensure_dataset_schema()
    with _db() as con: r=con.execute('SELECT COALESCE(SUM(size_bytes),0) n FROM datasets WHERE user_id=?',(user_id,)).fetchone()
    return int(r['n'])

def quota_info(user_id:str)->dict:
    used=usage(user_id); return {'used_bytes':used,'quota_bytes':QUOTA_BYTES,'remaining_bytes':max(0,QUOTA_BYTES-used),'percent':used/QUOTA_BYTES*100}

def _read_df(path:Path, fmt:str, limit:int|None=None)->pd.DataFrame:
    if fmt=='csv': return pd.read_csv(path,nrows=limit)
    if fmt=='jsonl':
        df=pd.read_json(path,lines=True); return df.head(limit) if limit else df
    if fmt=='json':
        df=pd.read_json(path); return df.head(limit) if limit else df
    if fmt=='parquet': return pd.read_parquet(path).head(limit) if limit else pd.read_parquet(path)
    raise ValueError('Unsupported dataset format')

def _fmt(filename:str)->str:
    low=filename.lower()
    if low.endswith('.csv'): return 'csv'
    if low.endswith('.jsonl') or low.endswith('.ndjson'): return 'jsonl'
    if low.endswith('.json'): return 'json'
    if low.endswith('.parquet'): return 'parquet'
    raise ValueError('Supported data files: .csv, .json, .jsonl/.ndjson, .parquet')

def save_upload(user_id:str, filename:str, data:bytes)->dict:
    ensure_dataset_schema(); size=len(data); used=usage(user_id)
    if size<=0: raise ValueError('File is empty')
    if used+size>QUOTA_BYTES: raise ValueError('This upload would exceed your 1 GB dataset quota')
    fmt=_fmt(filename); did=uuid4().hex[:12]; folder=DATA_ROOT/user_id; folder.mkdir(parents=True,exist_ok=True)
    safe=''.join(c for c in Path(filename).name if c.isalnum() or c in '._-')[:120] or 'dataset'
    path=folder/f'{did}_{safe}'; path.write_bytes(data)
    try:
        sample=_read_df(path,fmt,limit=5000); cols=len(sample.columns)
        # Counting all CSV rows cheaply; otherwise use parsed size when reasonable.
        if fmt=='csv':
            with path.open('rb') as f: rows=max(0,sum(1 for _ in f)-1)
        else: rows=len(_read_df(path,fmt)) if size<100*1024*1024 else None
    except Exception:
        path.unlink(missing_ok=True); raise
    row={'id':did,'user_id':user_id,'name':Path(filename).stem,'filename':filename,'path':str(path),'size_bytes':size,'format':fmt,'rows':rows,'columns':cols,'created_at':time.time(),'parent_id':None}
    with _db() as con: con.execute('INSERT INTO datasets VALUES(?,?,?,?,?,?,?,?,?,?,?)',tuple(row[k] for k in ['id','user_id','name','filename','path','size_bytes','format','rows','columns','created_at','parent_id']))
    return public(row)

def public(r)->dict:
    d=dict(r); d.pop('path',None); return d

def list_datasets(user_id:str)->list[dict]:
    ensure_dataset_schema()
    with _db() as con: rs=con.execute('SELECT * FROM datasets WHERE user_id=? ORDER BY created_at DESC',(user_id,)).fetchall()
    return [public(r) for r in rs]

def get_row(user_id:str,did:str):
    with _db() as con: r=con.execute('SELECT * FROM datasets WHERE id=? AND user_id=?',(did,user_id)).fetchone()
    if not r: raise KeyError(did)
    return r

def preview(user_id:str,did:str,limit:int=50)->dict:
    r=get_row(user_id,did); df=_read_df(Path(r['path']),r['format'],limit=max(1,min(limit,200)))
    return {'dataset':public(r),'columns':[{'name':str(c),'dtype':str(df[c].dtype)} for c in df.columns],'rows':json.loads(df.replace({np.nan:None}).to_json(orient='records'))}

def stats(user_id:str,did:str)->dict:
    r=get_row(user_id,did); df=_read_df(Path(r['path']),r['format'],limit=100_000)
    out=[]
    for c in df.columns:
        s=df[c]; base={'name':str(c),'dtype':str(s.dtype),'count':int(s.notna().sum()),'missing':int(s.isna().sum()),'unique':int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s):
            vals=pd.to_numeric(s,errors='coerce').dropna();
            if len(vals): base.update({'min':float(vals.min()),'max':float(vals.max()),'mean':float(vals.mean()),'std':float(vals.std(ddof=0))})
        else:
            base['top']=[{'value':str(k),'count':int(v)} for k,v in s.astype('string').value_counts(dropna=True).head(8).items()]
        out.append(base)
    return {'dataset':public(r),'columns':out,'sampled_rows':len(df)}

def chart(user_id:str,did:str,column:str,bins:int=20)->dict:
    r=get_row(user_id,did); df=_read_df(Path(r['path']),r['format'],limit=100_000)
    if column not in df: raise ValueError('Unknown column')
    s=df[column]
    if pd.api.types.is_numeric_dtype(s):
        vals=pd.to_numeric(s,errors='coerce').dropna().to_numpy()
        if not len(vals): return {'kind':'histogram','labels':[],'values':[]}
        counts,edges=np.histogram(vals,bins=max(5,min(bins,100)))
        labels=[f'{edges[i]:.3g}–{edges[i+1]:.3g}' for i in range(len(counts))]
        return {'kind':'histogram','column':column,'labels':labels,'values':counts.tolist()}
    vc=s.astype('string').value_counts(dropna=True).head(30)
    return {'kind':'bar','column':column,'labels':[str(x) for x in vc.index],'values':[int(x) for x in vc.values]}

def filter_rows(user_id:str,did:str,column:str,op:str,value:str,limit:int=200)->dict:
    r=get_row(user_id,did); df=_read_df(Path(r['path']),r['format'],limit=200_000)
    if column not in df: raise ValueError('Unknown column')
    s=df[column]
    if op in ('gt','gte','lt','lte'):
        x=pd.to_numeric(s,errors='coerce'); v=float(value); mask={'gt':x>v,'gte':x>=v,'lt':x<v,'lte':x<=v}[op]
    elif op=='eq': mask=s.astype(str)==str(value)
    elif op=='neq': mask=s.astype(str)!=str(value)
    elif op=='contains': mask=s.astype(str).str.contains(str(value),case=False,na=False)
    elif op=='missing': mask=s.isna()
    elif op=='not_missing': mask=s.notna()
    else: raise ValueError('Unsupported filter operator')
    out=df[mask].head(max(1,min(limit,500)))
    return {'matched':int(mask.sum()),'rows':json.loads(out.replace({np.nan:None}).to_json(orient='records'))}

def clean(user_id:str,did:str,ops:dict,name:str|None=None)->dict:
    r=get_row(user_id,did); df=_read_df(Path(r['path']),r['format'])
    if ops.get('drop_duplicates'): df=df.drop_duplicates()
    if ops.get('drop_missing'): df=df.dropna()
    fills=ops.get('fill_missing') or {}
    for c,v in fills.items():
        if c in df: df[c]=df[c].fillna(v)
    drops=ops.get('drop_columns') or []
    df=df.drop(columns=[c for c in drops if c in df],errors='ignore')
    rename=ops.get('rename_columns') or {}; df=df.rename(columns=rename)
    did2=uuid4().hex[:12]; folder=DATA_ROOT/user_id; path=folder/f'{did2}_{(name or r["name"]+"_cleaned")}.csv'; df.to_csv(path,index=False)
    size=path.stat().st_size
    if usage(user_id)+size>QUOTA_BYTES: path.unlink(missing_ok=True); raise ValueError('Cleaned copy would exceed your 1 GB quota')
    row={'id':did2,'user_id':user_id,'name':name or r['name']+'_cleaned','filename':path.name,'path':str(path),'size_bytes':size,'format':'csv','rows':len(df),'columns':len(df.columns),'created_at':time.time(),'parent_id':did}
    with _db() as con: con.execute('INSERT INTO datasets VALUES(?,?,?,?,?,?,?,?,?,?,?)',tuple(row[k] for k in ['id','user_id','name','filename','path','size_bytes','format','rows','columns','created_at','parent_id']))
    return public(row)

def delete(user_id:str,did:str):
    r=get_row(user_id,did); Path(r['path']).unlink(missing_ok=True)
    with _db() as con: con.execute('DELETE FROM datasets WHERE id=? AND user_id=?',(did,user_id))

def register_uploaded_path(user_id:str, filename:str, temp_path:Path, size:int)->dict:
    """Register a streamed upload without loading it all into memory."""
    ensure_dataset_schema(); used=usage(user_id)
    if size<=0: raise ValueError('File is empty')
    if used+size>QUOTA_BYTES: raise ValueError('This upload would exceed your 1 GB dataset quota')
    fmt=_fmt(filename); did=uuid4().hex[:12]; folder=DATA_ROOT/user_id; folder.mkdir(parents=True,exist_ok=True)
    safe=''.join(c for c in Path(filename).name if c.isalnum() or c in '._-')[:120] or 'dataset'
    path=folder/f'{did}_{safe}'; shutil.move(str(temp_path),str(path))
    try:
        sample=_read_df(path,fmt,limit=5000); cols=len(sample.columns)
        if fmt=='csv':
            with path.open('rb') as f: rows=max(0,sum(1 for _ in f)-1)
        else: rows=len(_read_df(path,fmt)) if size<100*1024*1024 else None
    except Exception:
        path.unlink(missing_ok=True); raise
    row={'id':did,'user_id':user_id,'name':Path(filename).stem,'filename':filename,'path':str(path),'size_bytes':size,'format':fmt,'rows':rows,'columns':cols,'created_at':time.time(),'parent_id':None}
    with _db() as con: con.execute('INSERT INTO datasets VALUES(?,?,?,?,?,?,?,?,?,?,?)',tuple(row[k] for k in ['id','user_id','name','filename','path','size_bytes','format','rows','columns','created_at','parent_id']))
    return public(row)
