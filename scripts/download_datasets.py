"""Download original simulation data and verify release SHA-256 checksums."""
from pathlib import Path
import argparse,hashlib,json,urllib.request

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',default=['fmri100','eeg100','voxel1000'])
    p.add_argument('--output-dir',type=Path,default=root/'data')
    args=p.parse_args()
    manifest=json.loads((root/'data/manifest.json').read_text(encoding='utf-8'))
    entries={x['dataset']:x for x in manifest['datasets']}
    unknown=set(args.datasets)-entries.keys()
    if unknown:p.error('Unknown datasets: '+', '.join(sorted(unknown)))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for key in args.datasets:
        item=entries[key];name=item['filename']
        if Path(name).name!=name:raise ValueError('Manifest filename must be a basename')
        target=args.output_dir/name
        if target.exists():
            if digest(target)!=item['sha256']:raise RuntimeError(f'{target} has a different checksum; move it aside before downloading.')
            print(f'{key}: already verified');continue
        temporary=target.with_suffix(target.suffix+'.partial')
        try:
            request=urllib.request.Request(item['url'],headers={'User-Agent':'SCDM-dataset-download'})
            with urllib.request.urlopen(request,timeout=120) as response,temporary.open('wb') as f:
                while chunk:=response.read(1024*1024):f.write(chunk)
            if temporary.stat().st_size!=item['bytes'] or digest(temporary)!=item['sha256']:
                raise RuntimeError(f'Checksum or size mismatch for {key}')
            temporary.replace(target);print(f'{key}: downloaded and verified')
        finally:
            if temporary.exists():temporary.unlink()

if __name__=='__main__':main()
