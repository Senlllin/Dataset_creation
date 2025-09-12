# pcdset

`pcdset` converts loose point cloud files into well organised datasets.
The first built in profile targets the [PCN dataset](https://github.com/wentaoyuan/pcn).
A second profile converts data into a [ShapeNet](https://shapenet.org/) style
layout.

## Installation

```bash
pip install -U pip
pip install -e .
```

## Generate an example manifest

```bash
pcdset example-manifest --profile pcn -o manifest.csv
```

## Convert using a manifest

```bash
pcdset convert \
  --profile pcn \
  --input D:/raw_points \
  --manifest D:/raw_points/manifest.csv \
  --out D:/datasets/PCN_custom \
  --partial-n 2048 --complete-n 16384 \
  --normalize unit --center --fps --dedup \
  --to-lmdb --lmdb-max-gb 64 \
  --workers 8
```

## Convert without manifest (directory inference)

```
<input>/
  partial/<cat>/<model>/<view>.(ply|pcd|csv|txt|npz)
  complete/<cat>/<model>.(ply|pcd|csv|txt|npz)
```

```bash
pcdset convert --profile pcn --input <input> --out <out> \
               --partial-n 2048 --complete-n 16384
```

## Validate a converted dataset

```bash
pcdset validate --profile pcn --root D:/datasets/PCN_custom
```

## ShapeNet profile

The ``shapenet`` profile converts arbitrary point clouds into a directory
structure compatible with ShapeNet style datasets.

### Example manifest

```bash
pcdset example-manifest --profile shapenet -o manifest.csv
```

### Convert using a manifest

```bash
pcdset convert \
  --profile shapenet \
  --input D:/raw_points \
  --manifest D:/raw_points/manifest.csv \
  --out D:/datasets/ShapeNet_custom \
  --points-n 2048 \
  --normalize unit --center --fps --dedup \
  --file-ext ply --basename points \
  --taxonomy-out taxonomy.csv \
  --to-lmdb --lmdb-max-gb 32 \
  --workers 8
```

### Convert by directory inference

```
<input>/<category>/<model_id>/*.(ply|pcd|txt|csv|npz)
```

```bash
pcdset convert --profile shapenet --input <input> --out <out> \
               --points-n 2048 --normalize unit --center
```

### Validate

```bash
pcdset validate --profile shapenet --root D:/datasets/ShapeNet_custom
```

## Quick start (Windows PowerShell)

```powershell
pip install -U pip
# Save all files into an empty folder
pip install -e .
pcdset list-profiles
pcdset example-manifest --profile pcn -o manifest.csv
pcdset convert --profile pcn --input D:\raw_points --manifest .\manifest.csv `
  --out D:\datasets\PCN_custom --partial-n 2048 --complete-n 16384 `
  --normalize unit --center --fps --dedup --to-lmdb --workers 8
pcdset validate --profile pcn --root D:\datasets\PCN_custom
```
