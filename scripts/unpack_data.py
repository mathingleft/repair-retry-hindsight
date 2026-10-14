"""Restore losslessly compressed experiment records using Python's standard library."""
import hashlib
import json
import lzma
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def local_path(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Invalid packed-data path')
    return path


def main():
    manifest = ROOT / 'manifests/packed_data.json'
    if not manifest.exists():
        return
    count = 0
    for row in json.loads(manifest.read_text()):
        src, dest = local_path(row['packed_path']), local_path(row['path'])
        if dest.exists():
            if digest(dest) != row['sha256']:
                raise ValueError(f'Existing data differ from frozen records: {dest}')
            continue
        if digest(src) != row['packed_sha256']:
            raise ValueError(f'Compressed data checksum mismatch: {src}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as output:
            temporary = Path(output.name)
            try:
                with lzma.open(src, 'rb') as stream:
                    shutil.copyfileobj(stream, output)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            if temporary.stat().st_size != row['bytes'] or digest(temporary) != row['sha256']:
                raise ValueError(f'Restored data checksum mismatch: {dest}')
            temporary.replace(dest)
        finally:
            temporary.unlink(missing_ok=True)
        count += 1
    print(f'Data ready: restored {count} files; all original bytes checked.', flush=True)


if __name__ == '__main__':
    main()
