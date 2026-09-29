from __future__ import annotations
from pathlib import Path
from uuid import uuid4
import json, sqlite3, time, os

ROOT = Path(os.environ.get('MODELFORGE_HOME', Path.home()/'.modelforge')).expanduser()
ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH = ROOT / 'modelforge.sqlite3'


def _db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript('''
    CREATE TABLE IF NOT EXISTS runs(
      id TEXT PRIMARY KEY, created_at REAL NOT NULL, model TEXT, name TEXT,
      payload TEXT NOT NULL, result TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS experiments(
      id TEXT PRIMARY KEY, created_at REAL NOT NULL, name TEXT NOT NULL,
      baseline_run TEXT, parent TEXT, graph_diff TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS experiment_runs(
      experiment_id TEXT NOT NULL, run_id TEXT NOT NULL,
      PRIMARY KEY(experiment_id, run_id)
    );
    CREATE TABLE IF NOT EXISTS imports(
      id TEXT PRIMARY KEY, created_at REAL NOT NULL, name TEXT NOT NULL,
      arch TEXT NOT NULL, path TEXT NOT NULL, metadata TEXT NOT NULL
    );
    ''')
    return con


def create_run(payload: dict, result: dict) -> dict:
    rid = uuid4().hex[:12]; ts=time.time()
    row={"id":rid,"created_at":ts,"payload":payload,"result":result}
    with _db() as con:
        con.execute('INSERT INTO runs VALUES(?,?,?,?,?,?)',(rid,ts,payload.get('model'),payload.get('name') or rid,json.dumps(payload),json.dumps(result)))
    return row


def list_runs() -> list[dict]:
    with _db() as con:
        rows=con.execute('SELECT id,created_at,model,name FROM runs ORDER BY created_at DESC').fetchall()
    return [dict(r) for r in rows]


def get_run(rid:str)->dict:
    with _db() as con:
        r=con.execute('SELECT * FROM runs WHERE id=?',(rid,)).fetchone()
    if not r: raise KeyError(rid)
    return {"id":r['id'],"created_at":r['created_at'],"payload":json.loads(r['payload']),"result":json.loads(r['result'])}


def create_experiment(name:str, baseline_run:str|None=None, parent:str|None=None, graph_diff:dict|None=None)->dict:
    eid=uuid4().hex[:12]; ts=time.time(); gd=graph_diff or {}
    with _db() as con:
        con.execute('INSERT INTO experiments VALUES(?,?,?,?,?,?)',(eid,ts,name,baseline_run,parent,json.dumps(gd)))
    return {"id":eid,"name":name,"baseline_run":baseline_run,"parent":parent,"graph_diff":gd,"runs":[],"created_at":ts}


def attach_run(eid:str,rid:str)->dict:
    with _db() as con:
        con.execute('INSERT OR IGNORE INTO experiment_runs VALUES(?,?)',(eid,rid))
        e=con.execute('SELECT * FROM experiments WHERE id=?',(eid,)).fetchone()
        rr=[x['run_id'] for x in con.execute('SELECT run_id FROM experiment_runs WHERE experiment_id=?',(eid,)).fetchall()]
    if not e: raise KeyError(eid)
    return {"id":e['id'],"name":e['name'],"baseline_run":e['baseline_run'],"parent":e['parent'],"graph_diff":json.loads(e['graph_diff']),"runs":rr,"created_at":e['created_at']}


def list_experiments()->list[dict]:
    with _db() as con:
        es=con.execute('SELECT * FROM experiments ORDER BY created_at DESC').fetchall()
        out=[]
        for e in es:
            rr=[x['run_id'] for x in con.execute('SELECT run_id FROM experiment_runs WHERE experiment_id=?',(e['id'],)).fetchall()]
            out.append({"id":e['id'],"name":e['name'],"baseline_run":e['baseline_run'],"parent":e['parent'],"graph_diff":json.loads(e['graph_diff']),"runs":rr,"created_at":e['created_at']})
    return out


def register_import(ref:str,name:str,arch:str,path:str,metadata:dict)->None:
    with _db() as con:
        con.execute('INSERT OR REPLACE INTO imports VALUES(?,?,?,?,?,?)',(ref,time.time(),name,arch,path,json.dumps(metadata)))


def get_import(ref:str)->dict|None:
    with _db() as con:
        r=con.execute('SELECT * FROM imports WHERE id=?',(ref,)).fetchone()
    if not r:return None
    return {"ref":r['id'],"name":r['name'],"arch":r['arch'],"path":r['path'],**json.loads(r['metadata'])}


def list_imports()->list[dict]:
    with _db() as con:
        rs=con.execute('SELECT * FROM imports ORDER BY created_at DESC').fetchall()
    return [{"ref":r['id'],"name":r['name'],"arch":r['arch'],"path":r['path'],**json.loads(r['metadata'])} for r in rs]


def compare_runs(a:dict,b:dict)->dict:
    oa=a['result'].get('output',{}); ob=b['result'].get('output',{})
    keys=['mean','std','min','max','l2','rms']; delta={}
    for k in keys:
        if isinstance(oa.get(k),(int,float)) and isinstance(ob.get(k),(int,float)):
            delta[k]={"a":oa[k],"b":ob[k],"delta":ob[k]-oa[k]}
    ca=a['result'].get('captures',{}); cb=b['result'].get('captures',{}); cap=[]
    for path in sorted(set(ca)&set(cb)):
        aa=ca[path].get('activation',{}); bb=cb[path].get('activation',{})
        if 'rms' in aa and 'rms' in bb: cap.append({"path":path,"rms_a":aa['rms'],"rms_b":bb['rms'],"delta":bb['rms']-aa['rms']})
    return {"run_a":a['id'],"run_b":b['id'],"output_delta":delta,"capture_delta":cap}
