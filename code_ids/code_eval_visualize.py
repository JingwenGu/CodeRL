import json
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ScriptArguments:
    evaluated_dir: Optional[str] = field(default="")
    beam: Optional[bool] = field(default=True)

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

with open(script_args.evaluated_dir,'r') as file:
    evaluated = json.load(file)

if script_args.beam:
    acc = [0,0]
    count = 0
    for problem in evaluated.values():
        if len(problem['canonical_solution'].split('\n')) <= 7:
            continue
        count += 1
        for traj in problem['result']:
            for step in traj:
                for res in step:
                    acc[1] += 1
                    if res.find('Accepted') != -1:
                        acc[0] += 1

    print(f'{count} problems with canonical solution length greater than 7')
    print(f'{acc[0]} accepted, {acc[1]} failed')
    print(f'accuracy={acc[0]/acc[1]}')
else:
    acc = [0,0]
    count = 0
    for problem in evaluated.values():
        if len(problem['canonical_solution'].split('\n')) <= 7:
            continue
        count += 1
        acc[1] += len(problem['result'])
        for res in problem['result']:
            if res.find('Accepted') != -1:
                acc[0] += 1

    print(f'{count} problems with canonical solution length greater than 7')
    print(f'{acc[0]} accepted, {acc[1]} failed')
    print(f'accuracy={acc[0]/acc[1]}')