# pcdset

`pcdset` converts loose point cloud files into well organised datasets.
The first built in profile targets the [PCN dataset](https://github.com/wentaoyuan/pcn).

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
