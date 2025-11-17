import os
import shutil

dir_name = "/root/Agents/llama2_inference/scales/moe_gate_trained_alternate_with_system_prompt_v2"
root, dirs, _ = next(os.walk(dir_name))
for d in dirs:
    _, _, files = next(os.walk(os.path.join(root,d+"/dialog")))
    print(d)
    print(len(files))
    if len(files)==0:
        shutil.rmtree(os.path.join(root,d))