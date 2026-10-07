from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

from inference import ROOT, Engine, Stream, dump


class Broker:
    def __init__(self, database, output):
        import duckdb
        self.db=duckdb.connect(str(database),read_only=True,config={'threads':1})
        self.output=output
        self.lock=threading.Lock()
        self.offset=0
        self.total=self.db.execute('SELECT count(*) FROM attempts').fetchone()[0]
        self.committed=0
        self.metrics={}
        self.started=time.time()

    def claim(self,count):
        with self.lock:
            start=self.offset
            self.offset=min(self.total,start+count)
            values=self.db.execute('''SELECT q.payload,a.sample_index,a.attempt_id,q.question_id,a.sampling_id
                FROM attempts a JOIN questions q USING(question_id)
                WHERE a.queue_id>=? AND a.queue_id<? ORDER BY a.queue_id''',[start,self.offset]).fetchall()
        return [dict(json.loads(row),sample_index=sample,attempt_id=identity,prefill_group_id=question,sampling_id=sampling) for row,sample,identity,question,sampling in values]

    def commit(self,count):
        with self.lock:
            self.committed+=count

    def progress(self,rank,value):
        with self.lock:
            self.metrics[str(rank)]=value
            active=[r for r in self.metrics.values() if r['status']=='RUNNING' and time.time()-r['timestamp']<120]
            status=dict(status='RUNNING',expected=self.total,completed=self.committed,claimed=self.offset,
                total_decode_tokens_per_second=sum(r.get('recent_decode_tokens_per_second',r.get('decode_tokens_per_second',0)) for r in active),
                total_useful_decode_tokens=sum(r.get('useful_decode_tokens',0) for r in self.metrics.values()),
                elapsed_seconds=time.time()-self.started,ranks=self.metrics)
            dump(self.output/'STATUS.json',status)


class Manager(BaseManager):
    pass


def prepare(args):
    import duckdb
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.model/'data/MANIFEST.json').read_text())
    database=output/'tasks.duckdb'
    db=duckdb.connect(str(database),config={'threads':args.cpu_workers})
    if args.suite=='vision':
        path=str(args.model/'data/vision/questions/*.jsonl')
        db.execute('CREATE TABLE raw AS SELECT *, id AS source_sample_id FROM read_json_auto(?, union_by_name=true)',[path])
    else:
        db.execute('CREATE TABLE raw AS SELECT * FROM read_parquet(?, union_by_name=true)',[str(args.model/f'data/{args.suite}/*.parquet')])
    counts=manifest[args.suite]['sources']
    db.execute('CREATE TABLE sampling(source VARCHAR,k INTEGER)')
    db.executemany('INSERT INTO sampling VALUES (?,?)',[(s,args.k if args.k is not None else
        (64 if n<100 else 16 if n<=10000 else 1) if args.suite=='text' else 1) for s,n in counts.items()])
    where='TRUE'
    parameters=[]
    if args.sources:
        where='source IN ('+','.join('?' for _ in args.sources)+')'
        parameters.extend(args.sources)
    limit=''
    if args.limit_per_source:
        limit=' QUALIFY row_number() OVER(PARTITION BY source ORDER BY source_sample_id)<=?'
        parameters.append(args.limit_per_source)
    db.execute('CREATE TABLE selected AS SELECT * FROM raw WHERE '+where+limit,parameters)
    if args.attempts:
        db.execute('CREATE TABLE requested AS SELECT * FROM read_parquet(?)',[str(args.attempts.resolve())])
        db.execute('DELETE FROM selected WHERE NOT EXISTS (SELECT 1 FROM requested r WHERE r.source=selected.source AND r.source_sample_id=selected.source_sample_id)')
    db.execute('''CREATE TABLE questions AS SELECT (row_number() OVER(ORDER BY source,source_sample_id)-1)::BIGINT question_id,
        source, source_sample_id, to_json(selected)::VARCHAR payload FROM selected''')
    if args.attempts:
        db.execute('''CREATE TABLE attempts AS SELECT (row_number() OVER(ORDER BY hash(q.question_id,r.sample_index,20260923),q.question_id,r.sample_index)-1)::BIGINT queue_id,
            q.question_id,r.sample_index::INTEGER sample_index,(q.question_id*64+r.sample_index)::BIGINT attempt_id,
            r.original_sampling_id::BIGINT sampling_id
            FROM questions q JOIN requested r USING(source,source_sample_id)''')
    else:
        db.execute('''CREATE TABLE attempts AS SELECT (row_number() OVER(ORDER BY hash(q.question_id,r.range,20260923),q.question_id,r.range)-1)::BIGINT queue_id,
            q.question_id,r.range::INTEGER sample_index,(q.question_id*64+r.range)::BIGINT attempt_id,(q.question_id*64+r.range)::BIGINT sampling_id
            FROM questions q JOIN sampling s USING(source), LATERAL range(s.k) r''')
    expected=db.execute('SELECT count(*) FROM attempts').fetchone()[0]
    count=db.execute('SELECT count(*) FROM questions').fetchone()[0]
    selected_counts=dict(db.execute('SELECT source,count(*) FROM questions GROUP BY source').fetchall())
    sampling=dict(db.execute('SELECT source,k FROM sampling').fetchall())
    db.close()
    protocol=dict(suite=args.suite,questions=count,expected=expected,source_counts=selected_counts,
        samples={s:sampling[s] for s in selected_counts},source_size_basis=counts,
        subset=args.limit_per_source is not None,k_override=args.k,gpus=args.gpus,batch=args.batch,
        profile=args.profile,max_new_tokens=args.max_new_tokens,seed=args.seed)
    if args.attempts:
        protocol.update(selected_attempts=str(args.attempts.resolve()),subset=True,
            sampling_identity='original_sampling_id from selected attempts; samples are not expanded again')
    dump(output/'PROTOCOL.json',protocol)
    return database,protocol


def evaluate(args):
    args.output=args.output.resolve()
    args.model=args.model.resolve()
    if args.suite=='memory':
        args.batch,args.profile,args.seed=1,'reference',2026091707
        args.max_new_tokens=min(args.max_new_tokens,1024)
    elif args.suite=='vision':
        args.batch,args.profile,args.seed=128,'reference',20260912
        args.max_new_tokens=min(args.max_new_tokens,32768)
    database,protocol=prepare(args)
    broker=Broker(database,args.output)
    Manager.register('broker',callable=lambda:broker)
    manager=Manager(address=('127.0.0.1',0),authkey=b'yanchor-evaluation')
    server=manager.get_server()
    port=server.address[1]
    threading.Thread(target=server.serve_forever,daemon=True).start()
    dump(args.output/'STATUS.json',dict(status='LOADING',expected=protocol['expected'],completed=0))
    started=time.time()
    def launch(pair):
        rank,gpu=pair
        cmd=[sys.executable,'-I',str(ROOT/'run.py'),'_worker','--rank',str(rank),'--port',str(port),
             '--suite',args.suite,'--model',str(args.model),'--output',str(args.output),'--batch',str(args.batch),
             '--profile',args.profile,'--max-new-tokens',str(args.max_new_tokens),'--seed',str(args.seed)]
        with (args.output/f'rank-{rank}.log').open('w') as log:
            return subprocess.run(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu},stdout=log,stderr=subprocess.STDOUT).returncode
    gpus=args.gpus.split(',')
    with ThreadPoolExecutor(len(gpus)) as pool:
        exits=list(pool.map(launch,enumerate(gpus)))
    if any(exits):
        dump(args.output/'STATUS.json',dict(status='FAILED',worker_exit_codes=exits,expected=broker.total,completed=broker.committed))
        raise SystemExit(1)
    result=dict(status='COMPLETE',expected=broker.total,observed=broker.committed,
        seconds=time.time()-started,ranks=broker.metrics)
    result['total_useful_decode_tokens']=sum(r.get('useful_decode_tokens',0) for r in broker.metrics.values())
    result['weighted_per_gpu_decode_tokens_per_second']=result['total_useful_decode_tokens']/max(1e-9,sum(r.get('decode_seconds',0) for r in broker.metrics.values()))
    dump(args.output/'GENERATION_COMPLETE.json',result)
    dump(args.output/'STATUS.json',result)
    if not args.generate_only:
        dump(args.output/'STATUS.json',dict(status='SCORING',expected=broker.total,generated=broker.committed))
        try:
            score(args)
        except Exception as error:
            dump(args.output/'STATUS.json',dict(status='FAILED',stage='scoring',error=str(error),
                expected=broker.total,generated=broker.committed))
            raise


def worker(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    Manager.register('broker')
    manager=Manager(address=('127.0.0.1',args.port),authkey=b'yanchor-evaluation')
    manager.connect()
    broker=manager.broker()
    destination=args.output/'generated'/f'rank-{args.rank}'
    destination.mkdir(parents=True,exist_ok=True)
    fragments=0
    def write(rows):
        nonlocal fragments
        if args.suite=='vision':
            rows=[dict(source=row['source'],attempt_id=row['attempt_id'],payload_json=json.dumps(row,ensure_ascii=False)) for row in rows]
        pq.write_table(pa.Table.from_pylist(rows),destination/f'{fragments:06d}.parquet',compression='zstd')
        fragments+=1
        broker.commit(len(rows))
    def report(value):
        broker.progress(args.rank,value)
    if args.suite=='vision':
        from vision import generate_visual
        metrics=generate_visual(args,broker.claim,write,report)
    else:
        engine=Engine(args.model,args.batch,args.profile)
        def fetch(count):
            rows=broker.claim(count)
            for row in rows:
                row['max_tokens']=args.max_new_tokens if args.suite=='text' else min(args.max_new_tokens,row.get('max_tokens',args.max_new_tokens))
            return engine.encode(rows)
        first=fetch(args.batch)
        if not first:
            report(dict(status='COMPLETE',timestamp=time.time(),completed=0))
            return
        stream=Stream(engine,fetch,write,report)
        settings=dict(cap=args.max_new_tokens,seed=args.seed)
        if args.suite=='memory':
            settings.update(temperature=1,top_k=1,top_p=1,presence_penalty=0,
                mechanical=False,stop_interval=32,sampling_backend='torch')
        metrics=engine.run_stream(stream,first,**settings)
    dump(args.output/f'rank-{args.rank}-metrics.json',metrics)
    report(dict(status='COMPLETE',timestamp=time.time(),completed=metrics.get('stream_committed_groups',0),
        useful_decode_tokens=metrics.get('decode_useful_token_events',0),decode_seconds=metrics.get('decode_seconds',0),
        decode_tokens_per_second=metrics.get('decode_tokens_per_second',0),metrics_file=f'rank-{args.rank}-metrics.json'))


_ENGINE=None
_IDS=None
_SUITE=None
_TOKENIZER=None
_ASSETS=None


def init_scoring(engine,suite,assets):
    global _ENGINE,_IDS,_SUITE,_TOKENIZER,_ASSETS
    _ENGINE,_SUITE,_ASSETS=engine,suite,Path(assets)
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(_ASSETS/'tokenizer.json'))
    _TOKENIZER=tok
    _IDS=(tok.encode('<think>',add_special_tokens=False).ids[0],tok.encode('</think>',add_special_tokens=False).ids[0])


def score_fragment(task):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from evaluation.fixed_protocol_score import score_one
    from evaluation.extension_metrics import score_prediction as extension
    from evaluation.visual_metrics import score_prediction as visual
    from evaluation.memory_metrics import score_generation as memory
    path,out=task
    records=pq.read_table(path,use_threads=False).to_pylist()
    results=[]
    for row in records:
        if _SUITE=='vision':
            row=json.loads(row['payload_json'])
            final=row['response'].rsplit('</think>',1)[1].split('<|im_end|>')[0].split('<|endoftext|>')[0].strip() if '</think>' in row['response'] else ''
            result={k:v for k,v in row.items() if k not in {'response','response_ids'}}
            result.update(visual(row,final))
        elif _SUITE=='memory':
            text=_TOKENIZER.decode(row['response_ids'],skip_special_tokens=True)
            result={k:v for k,v in row.items() if k not in {'response','response_ids','prompt_text'}}
            if result.get('answers') is None:
                result.pop('answers',None)
            result.update(memory(result,text))
        elif row['source'] in {'DROP','MuSR','LogiQA2'}:
            result={k:row.get(k) for k in ['source','source_sample_id','sample_index','subgroup','generated_tokens','finished_by_eos','cap_hit','finish_reason']}
            result.update(extension(row))
        else:
            metadata=json.loads(row['metadata_json'])
            if metadata.get('lcb_payload_path'):
                metadata['lcb_payload_path']=str(_ASSETS/metadata['lcb_payload_path'])
                row['metadata_json']=json.dumps(metadata,ensure_ascii=False)
            result=score_one(_ENGINE,row,*_IDS)
        results.append(result)
    keys=set().union(*(r.keys() for r in results))
    results=[{k:r.get(k) for k in keys} for r in results]
    if _SUITE=='vision':
        results=[dict(source=r['source'],payload_json=json.dumps(r,ensure_ascii=False)) for r in results]
    pq.write_table(pa.Table.from_pylist(results),out,compression='zstd')
    return len(results)


def score(args):
    import pyarrow.parquet as pq
    from evaluation import fixed_protocol_score as scoring
    from evaluation.visual_metrics import aggregate_scores
    protocol=json.loads((args.output/'PROTOCOL.json').read_text())
    suite=protocol['suite']
    destination=args.output/'scored'
    destination.mkdir(exist_ok=True)
    paths=sorted((args.output/'generated').glob('*/*.parquet'))
    tasks=[(p,destination/f'{i:07d}.parquet') for i,p in enumerate(paths)]
    config={'execution_contract':{'verifier_service':{
        'external_verifier_assets':str(ROOT/'runtime/third_party/verifiers'),
        'external_timeout_seconds':180,'math_workers':32,'code_workers':16,'instruction_workers':8}}}
    context=mp.get_context('fork')
    started=time.time()
    with scoring.shared_scoring_engine(config,context) as engine:
        with ProcessPoolExecutor(args.cpu_workers,mp_context=context,initializer=init_scoring,initargs=(engine,suite,args.model)) as pool:
            count=sum(pool.map(score_fragment,tasks))
    with ThreadPoolExecutor(args.cpu_workers) as pool:
        parts=list(pool.map(lambda p:pq.read_table(p,use_threads=False).to_pylist(),sorted(destination.glob('*.parquet'))))
    rows=[r for part in parts for r in part]
    if suite=='vision':
        rows=[json.loads(r['payload_json']) for r in rows]
    grouped=defaultdict(list)
    for r in rows:
        grouped[r['source']].append(r)
    sources={}
    if suite=='vision':
        sources=aggregate_scores(rows)
        for source,values in grouped.items():
            sources[source].update(mean_generated_tokens=statistics.fmean(r['generated_tokens'] for r in values),
                cap_hit_rate=statistics.fmean(r['cap_hit'] for r in values),
                eos_rate=statistics.fmean(r['finished_by_eos'] for r in values),
                extraction_failure_rate=statistics.fmean(r['extraction_failed'] for r in values))
    else:
        for source,values in sorted(grouped.items()):
            if suite=='text' and source not in {'DROP','MuSR','LogiQA2'}:
                sources[source]=scoring.summarize(values)
            elif suite=='text':
                if source=='MuSR':
                    subsets=defaultdict(list)
                    for row in values:subsets[row['subgroup']].append(row['score'])
                    sources[source]=dict(score=statistics.fmean(statistics.fmean(v) for v in subsets.values()),
                        subset_scores={k:statistics.fmean(v) for k,v in subsets.items()})
                else:
                    sources[source]={'score':statistics.fmean(r['score'] for r in values)}
                if source=='DROP':
                    sources[source].update(exact_match=statistics.fmean(r['exact_match'] for r in values),f1=sources[source]['score'])
            else:
                sources[source]={k:statistics.fmean(r[k] for r in values)
                    for k in ['exact_match','all_queries_correct','parsed_query_fraction']}
                families=defaultdict(list)
                pairs=defaultdict(dict)
                for row in values:
                    families[row['task_family']].append(row)
                    pairs[row['pair_id']][row['variant']]=row
                sources[source]['tasks']={name:dict(contexts=len(part),
                    query_accuracy=statistics.fmean(r['exact_match'] for r in part),
                    all_four_correct=statistics.fmean(r['all_queries_correct'] for r in part))
                    for name,part in families.items()}
                paired=defaultdict(list)
                for pair in pairs.values():
                    if 'original' in pair and 'cf' in pair:
                        a,b=pair['original'],pair['cf']
                        for q in a['answers']:
                            key='answer_changed' if a['answers'][q]!=b['answers'][q] else 'answer_stable'
                            paired[key].append(a['query_correct'][q] and b['query_correct'][q])
                sources[source]['paired_queries']={name:dict(pairs=len(part),both_correct=statistics.fmean(part))
                    for name,part in paired.items()}
            sources[source].update(attempts=len(values),mean_generated_tokens=statistics.fmean(r['generated_tokens'] for r in values),
                cap_hit_rate=statistics.fmean(r['cap_hit'] for r in values),eos_rate=statistics.fmean(r['finished_by_eos'] for r in values))
    complete=count==protocol['expected']
    result=dict(status='COMPLETE' if complete else 'INCOMPLETE',suite=suite,expected=protocol['expected'],observed=count,
        sources=sources,seconds=time.time()-started,protocol=protocol,cpu_verifier_revision=scoring.CPU_VERIFIER_REVISION)
    dump(args.output/'RESULTS.json',result)
    dump(args.output/'STATUS.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
