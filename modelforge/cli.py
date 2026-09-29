from __future__ import annotations
import argparse, json

def main():
    p=argparse.ArgumentParser(prog='modelforge',description='ModelForge research workbench')
    sub=p.add_subparsers(dest='cmd')
    serve=sub.add_parser('serve',help='start the local research workbench'); serve.add_argument('--host',default='127.0.0.1'); serve.add_argument('--port',type=int,default=8000); serve.add_argument('--reload',action='store_true')
    stitch=sub.add_parser('stitch',help='run the ResNet stitching smoke test'); stitch.add_argument('--source',default='resnet18'); stitch.add_argument('--source-cut',default='layer2'); stitch.add_argument('--target',default='resnet34'); stitch.add_argument('--target-entry',default='layer3'); stitch.add_argument('--image-size',type=int,default=224); stitch.add_argument('--steps',type=int,default=1)
    args=p.parse_args()
    if args.cmd in (None,'serve'):
        import uvicorn; uvicorn.run('modelforge.api:app',host=getattr(args,'host','127.0.0.1'),port=getattr(args,'port',8000),reload=getattr(args,'reload',False)); return
    if args.cmd=='stitch':
        from .models import load_model
        from .stitch import build_hybrid
        from .train import synthetic_train
        src,tgt=load_model(args.source),load_model(args.target); hybrid,plan=build_hybrid(src,args.source_cut,tgt,args.target_entry,args.image_size); result=synthetic_train(hybrid,steps=args.steps,batch_size=1,image_size=args.image_size); print(json.dumps({'plan':plan.to_dict(),'training':result.to_dict()},indent=2))
if __name__=='__main__':main()
