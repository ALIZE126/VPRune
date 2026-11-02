"""Package only explicitly allowlisted public source files; no Git history."""
import argparse
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=root / 'dist/vprune-source.zip')
    args = parser.parse_args()
    files = [root / p for p in ['README.md', 'LICENSE', 'NOTICE', '.gitignore', 'pyproject.toml']]
    for folder in ['vprune', 'tests', 'scripts', 'examples']:
        files.extend((root / folder).glob('*.py'))
    files.extend((root / '.github/workflows').glob('*.yml'))
    if (root / 'examples/demo.png').is_file():
        files.append(root / 'examples/demo.png')
    for file in files:
        if not file.is_file() or file.is_symlink():
            raise ValueError(f'Missing file or unsupported symlink: {file.name}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as z:
        for file in sorted(files):
            z.write(file, Path('vprune') / file.relative_to(root))
    with zipfile.ZipFile(args.output) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist())) == len(files)
    print(f'{args.output}: {len(files)} files, {args.output.stat().st_size} bytes')


if __name__ == '__main__':
    main()
