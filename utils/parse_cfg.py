

def parse_config(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [l for l in lines if l and not l.startswith('#')]
    lines = [l.rstrip().lstrip() for l in lines]
    block_defs = []
    for line in lines:
        if line.startswith('['): # start of a new block
            block_defs.append({})
            block_defs[-1]['type'] = line[1:-1].rstrip()
        else:
            k, v = line.split('=')
            block_defs[-1][k.rstrip()] = v.lstrip()
    return block_defs

def parse_data_config(path):
    options = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        k, v = line.split('=')
        options[k.strip()] = v.strip()
    return options
