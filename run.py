#!/usr/bin/env python3
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runtime'))
os.environ.pop('PYTHONPATH', None)

import argparse
import json
import runpy
import time

from inference import ROOT, Engine, dump


def benchmark(args):
    import torch
    from model.inference_kernels import cuda_graph_generate_top20_dynamic_refill
    engine = Engine(args.model, args.batch, args.profile, backend='cuda')
    groups = args.batch // args.k
    positions = torch.arange(args.prefill).unsqueeze(0)
    group_ids = torch.arange(groups).unsqueeze(1)
    unique = (positions * 104729 + 8191 + group_ids * 15485863).remainder(248320 - 4096) + 2048
    ids = unique.repeat_interleave(args.k, dim=0).cuda()
    mask = torch.ones_like(ids)
    common = dict(slot_batch_size=args.batch, max_new_tokens=args.max_new_tokens,
        temperature=1.0, top_k=20, top_p=0.95, presence_penalty=1.5,
        eos_token_ids=(), pad_token_id=engine.tokenizer.pad_token_id, stop_check_interval_tokens=128,
        attempt_ids=torch.arange(args.batch, device='cuda'),
        prefill_group_ids=torch.arange(args.batch, device='cuda').div(args.k, rounding_mode='floor'),
        refill_batch_size=min(args.batch,16),refill_prefetch_size=min(args.batch,16),
        prefill_chunk_size=min(4 if args.prefill>=8192 else 16,max(1,32768//args.prefill)),
        prefill_token_chunk_size=16384, sampling_vocab_size=len(engine.tokenizer),
        counter_sampling_backend='triton_top20', adaptive_tail_compaction=False,
        mechanical_repetition_early_stop=False)
    with torch.inference_mode():
        started=time.perf_counter()
        warm=cuda_graph_generate_top20_dynamic_refill(engine.model,ids,mask,**common,counter_seed=20260827,
            max_new_tokens_per_attempt=torch.full((args.batch,),args.warmup,device='cuda',dtype=torch.long))
        cold=dict(seconds=time.perf_counter()-started,ttft_seconds=warm.metrics['initial_prefill_seconds'])
        del warm
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result=cuda_graph_generate_top20_dynamic_refill(engine.model,ids,mask,**common,
            counter_seed=20260828+args.prefill+(args.batch if args.batch>1 else 0))
    metrics=result.metrics
    prefill=metrics['initial_prefill_seconds']+metrics['refill_prefill_seconds']
    decode=metrics['generation_wall_seconds']-prefill
    receipt=dict(status='COMPLETE',batch=args.batch,prefill_tokens=args.prefill,
        generated_tokens=int(result.generated_lengths.sum()),decode_tokens=args.max_new_tokens,
        ttft_seconds=metrics['initial_prefill_seconds'],decode_seconds=decode,
        decode_tokens_per_second=metrics['decode_useful_token_events']/decode,
        prefill_unique_tokens_per_second=groups*args.prefill/prefill,
        end_to_end_output_tokens_per_second=int(result.generated_lengths.sum())/metrics['generation_wall_seconds'],
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        warmup=cold,model_load_seconds=engine.load_seconds,operators=engine.operators,
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),metrics=metrics)
    dump(args.output,receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


def serve(args):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from queue import Queue
    import threading
    import uuid
    from transformers import TextStreamer
    engine = Engine(args.model, args.batch, args.profile, backend=args.backend, device=args.device, dtype=args.dtype)
    warmup_started = time.perf_counter()
    if args.batch == 1:
        engine.generate([dict(prompt='Compute 17 × 23.', max_tokens=16)], cap=16, mechanical=False)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, status=200):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/v1/models':
                self.send_json(dict(object='list', data=[dict(id='YANchor-4B', object='model', owned_by='RocoreMatrix')]))
            else:
                self.send_json(dict(status='READY', model='YANchor-4B', backend=engine.backend, batch_capacity=args.batch))

        def do_POST(self):
            try:
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                standard = self.path == '/v1/chat/completions'
                if self.path not in {'/', '/v1/chat/completions'}:
                    self.send_json({'error': {'message': 'Unknown endpoint'}}, 404)
                    return
                unsupported = [key for key in ['tools', 'tool_choice', 'response_format', 'stop'] if request.get(key)]
                if unsupported or request.get('n', 1) != 1:
                    self.send_json({'error': {'message': 'This endpoint supports plain text chat with n=1.'}}, 400)
                    return
                prompts = [{'messages': request['messages']}] if standard else request.get('prompts') or [{'messages': request['messages']}]
                settings = {key: request[key] for key in ['temperature', 'top_k', 'top_p', 'presence_penalty', 'seed'] if key in request}
                cap = int(request.get('max_completion_tokens', request.get('max_tokens', request.get('max_new_tokens', args.max_new_tokens))))
                identity, created = 'chatcmpl-' + uuid.uuid4().hex, int(time.time())

                def completion(rows):
                    row = rows[0]
                    text = engine.tokenizer.decode([t for t in row['response_ids'] if t not in (248044, 248046)], skip_special_tokens=False)
                    usage = dict(prompt_tokens=row['prompt_tokens'], completion_tokens=row['generated_tokens'], total_tokens=row['prompt_tokens']+row['generated_tokens'])
                    reason = 'length' if row['cap_hit'] else 'stop'
                    return text, usage, reason

                if standard and request.get('stream'):
                    queue = Queue()
                    class TokenStream(TextStreamer):
                        def on_finalized_text(self, text, stream_end=False):
                            if text:
                                queue.put(('text', text))
                    decoder = TokenStream(engine.tokenizer, skip_prompt=False, skip_special_tokens=False)
                    def on_tokens(attempt, tokens):
                        tokens = [t for t in tokens if t not in (248044, 248046)]
                        if tokens:
                            decoder.put(engine.torch.tensor(tokens)[None])
                    def generate():
                        try:
                            with lock:
                                rows, metrics = engine.generate(prompts, cap=cap, on_tokens=on_tokens, **settings)
                            decoder.end()
                            queue.put(('done', completion(rows)))
                        except Exception as error:
                            queue.put(('error', str(error)))
                    threading.Thread(target=generate, daemon=True).start()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    def event(value):
                        self.wfile.write(('data: ' + json.dumps(value, ensure_ascii=False) + '\n\n').encode())
                        self.wfile.flush()
                    event(dict(id=identity, object='chat.completion.chunk', created=created, model='YANchor-4B', choices=[dict(index=0, delta={'role':'assistant'}, finish_reason=None)]))
                    while True:
                        kind, value = queue.get()
                        if kind == 'error':
                            event({'error': {'message': value}})
                            break
                        if kind == 'text':
                            event(dict(id=identity, object='chat.completion.chunk', created=created, model='YANchor-4B', choices=[dict(index=0, delta={'content':value}, finish_reason=None)]))
                        else:
                            text, usage, reason = value
                            event(dict(id=identity, object='chat.completion.chunk', created=created, model='YANchor-4B', choices=[dict(index=0, delta={}, finish_reason=reason)]))
                            if request.get('stream_options', {}).get('include_usage'):
                                event(dict(id=identity, object='chat.completion.chunk', created=created, model='YANchor-4B', choices=[], usage=usage))
                            break
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                    return
                with lock:
                    rows, metrics = engine.generate(prompts, cap=cap, **settings)
                if standard:
                    text, usage, reason = completion(rows)
                    self.send_json(dict(id=identity, object='chat.completion', created=created, model='YANchor-4B', choices=[dict(index=0, message={'role':'assistant', 'content':text}, finish_reason=reason)], usage=usage))
                else:
                    self.send_json(dict(outputs=rows, metrics=metrics))
            except (KeyError, TypeError, ValueError) as error:
                self.send_json({'error': {'message': str(error)}}, 400)
            except (BrokenPipeError, ConnectionResetError):
                return

    print(json.dumps(dict(status='READY', host=args.host, port=args.port, backend=engine.backend,
        batch=args.batch, warmup_seconds=time.perf_counter()-warmup_started)), flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


def main():
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('NLTK_DATA', str(ROOT/'runtime/third_party/verifiers/nltk_data'))
    if len(sys.argv)>1 and sys.argv[1]=='_verify':
        kind=sys.argv[2]
        sys.argv=[sys.argv[0],*sys.argv[3:]]
        runpy.run_module('evaluation.'+{'math':'math_verifier','code':'code_verifier','instruction':'instruction_verifier'}[kind],run_name='__main__')
        return
    parser=argparse.ArgumentParser(description='YANchor-4B inference and reproducible evaluation')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['demo','generate','serve','benchmark','evaluate','score','status','_worker']:
        p=sub.add_parser(name)
        p.add_argument('--model',type=Path,default=ROOT)
        p.add_argument('--output',type=Path,default=Path('outputs')/name)
        if name in {'demo','generate','serve','benchmark','evaluate','_worker'}:
            p.add_argument('--batch',type=int,default=1 if name in {'demo','serve'} else 320)
            p.add_argument('--backend',choices=['auto','torch','cuda'],default='auto')
            p.add_argument('--device')
            p.add_argument('--dtype',choices=['float32','float16','bfloat16'])
            p.add_argument('--profile',choices=['fast','reference'],default='fast')
            p.add_argument('--max-new-tokens',type=int,default=4096 if name in {'demo','benchmark'} else 131072)
        if name in {'demo','generate','evaluate','_worker'}:
            p.add_argument('--seed',type=int,default=20260923)
        if name=='demo':
            p.add_argument('--prompt',default='计算 17 × 23，并简短说明计算过程。')
        if name=='generate':
            p.add_argument('--input',type=Path,required=True)
        if name=='serve':
            p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8000)
        if name=='benchmark':
            p.add_argument('--prefill',type=int,default=4096)
            p.add_argument('--warmup',type=int,default=256)
            p.add_argument('--k',type=int,default=None)
        if name in {'evaluate','score'}:
            p.add_argument('--suite',choices=['text','memory','vision'],default='text')
            p.add_argument('--sources',nargs='*')
            p.add_argument('--cpu-workers',type=int,default=160)
        if name=='evaluate':
            p.add_argument('--gpus',default='0')
            p.add_argument('--attempts',type=Path,help='Rerun selected source/question/sample identities from a Parquet file.')
            p.add_argument('--limit-per-source',type=int)
            p.add_argument('--k',type=int,help='Override automatic K; such runs are reported separately.')
            p.add_argument('--generate-only',action='store_true')
        if name=='_worker':
            p.add_argument('--rank',type=int,required=True)
            p.add_argument('--port',type=int,required=True)
            p.add_argument('--suite',choices=['text','memory','vision'],default='text')
    args=parser.parse_args()
    if args.command=='benchmark':
        args.k=args.k or (1 if args.batch==1 else 8)
        benchmark(args)
    elif args.command=='serve':
        serve(args)
    elif args.command in {'demo','generate'}:
        prompts=[args.prompt] if args.command=='demo' else [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
        engine=Engine(args.model,args.batch,args.profile,backend=args.backend,device=args.device,dtype=args.dtype)
        rows,metrics=engine.generate(prompts,cap=args.max_new_tokens,seed=args.seed)
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'responses.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        dump(args.output/'metrics.json',metrics)
        for row in rows:
            print(row['response'],flush=True)
        print(json.dumps({'ttft_seconds':metrics['initial_prefill_seconds'],'decode_tokens_per_second':metrics['decode_tokens_per_second'],'output':str(args.output)},ensure_ascii=False))
    elif args.command=='status':
        print((args.output/'STATUS.json').read_text())
    else:
        from evaluate import evaluate,score,worker
        {'evaluate':evaluate,'score':score,'_worker':worker}[args.command](args)

if __name__ == '__main__':
    main()
