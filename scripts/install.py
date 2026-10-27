"""Install either/both independent SMoPE extensions into official repository.

Safe: refuses unknown upstream source layouts and creates backups.
"""
import argparse
import pathlib
import shutil
import sys

ROOT=pathlib.Path(__file__).resolve().parents[1]


def replace_once(src, old, new, target):
    if src.count(old)!=1:
        raise RuntimeError(f'Upstream layout unexpected in {target}: found {src.count(old)} anchors, expected one')
    return src.replace(old,new,1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=pathlib.Path,required=True)
    p.add_argument('--task',choices=['1','2','both'],default='both')
    args=p.parse_args()
    repo=args.repo.resolve()
    vit=repo/'models/vit.py'; learner=repo/'learners/prompt.py'
    for f in ([vit] if args.task=='1' else [learner] if args.task=='2' else [vit,learner]):
        if not f.is_file(): raise SystemExit(f'Missing official SMoPE source: {f}')
    patches={}
    if args.task in ('1','both'):
        src=vit.read_text()
        if 'SMOPE_ROUTING' in src: raise SystemExit('Task 1 patch already present. Restore original backup first.')
        start='        if topk > 0 and prompt is not None and reduce_query:\n'
        replacement='''        if topk > 0 and prompt is not None and reduce_query and os.environ.get("SMOPE_ROUTING", "input") == "expert":
            # In expert-choice every expert chooses capacity samples per head.
            eps = eps_decay.unsqueeze(0).unsqueeze(2)  # [1,H,1,E]
            with torch.no_grad():
                score_span = prompt_score_.amax(-1, keepdim=True) - prompt_score_.amin(-1, keepdim=True)
            scores = prompt_score_ - eps * score_span
            prompt_score_label_ = scores
            capacity_factor = float(os.environ.get("SMOPE_EC_CAPACITY_FACTOR", "1.0"))
            chosen, _ = expert_choice_mask(scores.detach(), topk, capacity_factor)
            pk_selected, pv_selected, valid = pack_selected_prompts(pk, pv, chosen)
            v_ = torch.cat((pv_selected, v), dim=2)
            prompt_logits = (q_prompt @ pk_selected.transpose(-2,-1)) * self.scale
            prompt_logits = prompt_logits.expand(-1,-1,N,-1)
            prompt_logits = prompt_logits.masked_fill(~valid.expand(-1,-1,N,-1), torch.finfo(prompt_logits.dtype).min)
            attn = torch.cat((prompt_logits, attn[:,:,:,prompt_length:]), dim=-1)
            if os.environ.get("SMOPE_EC_TRACE", "0") == "1":
                import json
                log = utilization_report(chosen)
                with open(os.environ.get("SMOPE_EC_TRACE_PATH", "expert_activations.jsonl"),"a") as f:
                    f.write(json.dumps(log)+"\\n")
        elif topk > 0 and prompt is not None and reduce_query:
'''
        src=replace_once(src,start,replacement,vit)
        src='from task1.expert_choice import expert_choice_mask, pack_selected_prompts, utilization_report\nimport os\n'+src
        patches[vit]=src
    if args.task in ('2','both'):
        src=learner.read_text()
        if 'SMOPE_ORTHO_LAMBDA' in src: raise SystemExit('Task 2 patch already present. Restore original backup first.')
        # Only OnePrompt.update_model, not Prompt.update_model or other learners.
        marker='class OnePrompt(Prompt):'
        if src.count(marker)!=1: raise RuntimeError('Cannot locate OnePrompt learner')
        before,after=src.split(marker,1)
        old='        total_loss = total_loss + prompt_loss.sum()\n'
        new='''        total_loss = total_loss + prompt_loss.sum()
        # Task 2: penalize squared off-diagonal cosine similarities of
        # trainable prefix keys. Disabled by default: exact baseline path.
        ortho_weight = float(os.environ.get("SMOPE_ORTHO_LAMBDA", "0"))
        if ortho_weight < 0:
            raise ValueError("SMOPE_ORTHO_LAMBDA must be nonnegative")
        if ortho_weight > 0:
            from task2.orthogonal import orthogonal_loss
            prompt_module = self.model.module.prompt if hasattr(self.model, "module") else self.model.prompt
            total_loss = total_loss + ortho_weight * orthogonal_loss(prompt_module)
'''
        # Locate only update_model in OnePrompt and ensure loss is there.
        a=after.index('    def update_model('); b=after.index('    def learn_prompt(',a)
        part=replace_once(after[a:b],old,new,learner)
        after=after[:a]+part+after[b:]
        patches[learner]=before+marker+after
    # Preflight compile all patches before touching source files.
    for target,source in patches.items(): compile(source,str(target),'exec')
    # Backups and installation; all paths now validated.
    for target in patches:
        backup=target.with_suffix(target.suffix+'.before_dual_tasks')
        if backup.exists(): raise SystemExit(f'Backup already exists: {backup}; refusing overwrite')
    for target,source in patches.items():
        shutil.copy2(target,target.with_suffix(target.suffix+'.before_dual_tasks'))
        target.write_text(source)
        print('Patched:',target)
    for folder in (['task1'] if args.task=='1' else ['task2'] if args.task=='2' else ['task1','task2']):
        dst=repo/folder; dst.mkdir(exist_ok=True)
        shutil.copy2(ROOT/folder/('__init__.py'),dst/'__init__.py')
        shutil.copy2(ROOT/folder/('expert_choice.py' if folder=='task1' else 'orthogonal.py'),dst/('expert_choice.py' if folder=='task1' else 'orthogonal.py'))
    print('Installed. Default remains original SMoPE; set environment variables to enable tasks.')

if __name__=='__main__':main()
