from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile
import torch
from safetensors.torch import load_file
from transformers import AutoConfig,AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model,Qwen3_5VisionModel

from inference import Engine,Stream


class VisualModel(torch.nn.Module):
    get_rope_index=Qwen3_5Model.get_rope_index
    get_vision_position_ids=Qwen3_5Model.get_vision_position_ids
    get_placeholder_mask=Qwen3_5Model.get_placeholder_mask

    def __init__(self,text,config,visual):
        super().__init__()
        self.text,self.config,self.visual=text,config,visual

    def get_input_embeddings(self):
        return self.text.get_input_embeddings()

    def forward(self,input_ids,attention_mask,pixel_values,image_grid_thw):
        embeds=self.get_input_embeddings()(input_ids)
        features=self.visual(pixel_values.to(dtype=embeds.dtype),grid_thw=image_grid_thw,return_dict=True).pooler_output
        masks=self.get_placeholder_mask(input_ids,embeds,image_features=features)
        embeds=embeds.masked_scatter(masks[0],features.to(embeds))
        rope,deltas=self.get_rope_index(input_ids,mm_token_type_ids=(input_ids==self.config.image_token_id).long(),
            image_grid_thw=image_grid_thw,attention_mask=attention_mask)
        physical=(attention_mask.long().cumsum(-1)-1).clamp_min(0)
        positions=torch.cat((physical.unsqueeze(0),rope),dim=0)
        output=self.text(inputs_embeds=embeds,attention_mask=attention_mask,position_ids=positions,use_cache=True,logits_to_keep=1)
        lengths=attention_mask.sum(-1)
        spatial=lengths+deltas.flatten()
        output['vision_decode_positions']=torch.stack((lengths,spatial,spatial,spatial))[:,:,None]
        return output


class VisualContext:
    def __init__(self,model,stream,batch):
        self.model,self.stream=model,stream
        self.positions=torch.empty((4,2*batch+1,1),dtype=torch.long,device='cuda')

    def __call__(self,ids,mask,indices):
        selected=[self.stream.rows[i]['vision_inputs'] for i in indices.tolist()]
        output=self.model(input_ids=ids,attention_mask=mask,
            pixel_values=torch.cat([v['pixel_values'] for v in selected]).to(ids.device),
            image_grid_thw=torch.cat([v['image_grid_thw'] for v in selected]).to(ids.device))
        self.positions.index_copy_(1,indices,output['vision_decode_positions'])
        return output

    def decode_positions(self,indices):
        return self.positions.index_select(1,indices)


def generate_visual(args,claim,write,report):
    engine=Engine(args.model,args.batch,args.profile)
    config=AutoConfig.from_pretrained(args.model/'vision',local_files_only=True)
    config.vision_config._attn_implementation='flash_attention_2'
    visual=Qwen3_5VisionModel(config.vision_config)
    visual.load_state_dict(load_file(str(args.model/'vision/model.safetensors')))
    model=VisualModel(engine.model,config,visual).cuda().to(dtype=torch.bfloat16).eval().requires_grad_(False)
    processor=AutoProcessor.from_pretrained(args.model/'vision',local_files_only=True)
    processor.tokenizer.padding_side='left'
    processor.image_processor.size={'longest_edge':4194304,'shortest_edge':65536}
    media=ZipFile(args.model/'data/vision/media.zip')
    pool=ThreadPoolExecutor(20)
    def image(row):
        from PIL import Image
        with media.open(row['image_path']) as source, Image.open(source) as image:
            return processor.image_processor(images=image.convert('RGB'),return_tensors='pt')
    def fetch(count):
        rows=claim(count)
        if not rows:return []
        images=list(pool.map(image,rows))
        texts=[]
        for row,pixels in zip(rows,images):
            question=((row.get('hint') or '')+'\n'+row['question']).strip()
            choices={k:v for k,v in (row.get('choices') or {}).items() if v is not None}
            row['choices']=choices
            if choices:
                question+='\n'+'\n'.join(f'{k}. {v}' for k,v in choices.items())
                question+='\nGive only the option letter as your final answer.'
            elif row['source'] in ('mme','pope'):
                question+='\nGive only Yes or No as your final answer.'
            else:question+='\nGive a short answer without explanation as your final answer.'
            text=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':question}]}],
                tokenize=False,add_generation_prompt=True,enable_thinking=True)
            count=int(pixels['image_grid_thw'].prod())//processor.image_processor.merge_size**2
            texts.append(text.replace(processor.image_token,processor.image_token*count,1))
        ids=processor.tokenizer(texts,add_special_tokens=False,padding=False)['input_ids']
        return [dict(row,input_ids=token,vision_inputs=pixels,max_tokens=args.max_new_tokens)
                for row,token,pixels in zip(rows,ids,images)]
    first=fetch(args.batch)
    if not first:return {}
    stream=Stream(engine,fetch,write,report)
    stream.prefill_chunk_size=lambda width:4
    context=VisualContext(model,stream,args.batch)
    try:
        return engine.run_stream(stream,first,cap=args.max_new_tokens,temperature=1,top_k=20,top_p=.95,
            presence_penalty=1.5,seed=args.seed,mechanical=False,prefill_context=context,
            eos_ids=(processor.tokenizer.eos_token_id,))
    finally:
        pool.shutdown(wait=True)
        media.close()
