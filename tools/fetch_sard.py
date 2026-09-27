#!/usr/bin/env python3
"""Selectively extract members from the SARD Zenodo archives.

The ComplementarySet archive is 35 GB but the nine videos inside it range from
100 MB to 12 GB. Zenodo serves HTTP byte ranges, so individual members can be
pulled without downloading the whole archive. Reading the zip central directory
costs a few hundred KB; each member then costs exactly its compressed size.

    python3 tools/fetch_sard.py --list complementary
    python3 tools/fetch_sard.py complementary 7.mp4 7.txt --out data/external/sard

Dataset: https://zenodo.org/records/19926448  (CC-BY-4.0)
"""

import argparse
import os
import struct
import subprocess
import sys
import zlib

from remotezip import RemoteZip

ARCHIVES = {
    "core": "https://zenodo.org/records/19926448/files/CoreSet.zip?download=1",
    "complementary": (
        "https://zenodo.org/records/19926448/files/ComplementarySet.zip?download=1"
    ),
}

CHUNK = 1 << 20


def index(url):
    """Map basename -> (header_offset, compressed_size, uncompressed_size)."""
    with RemoteZip(url) as z:
        return {
            os.path.basename(i.filename): (
                i.header_offset,
                i.compress_size,
                i.file_size,
                i.compress_type,
            )
            for i in z.infolist()
            if not i.is_dir()
        }


def curl_range(url, start, length, dest):
    """Fetch bytes [start, start+length) into dest, resuming a partial dest.

    Resume is done by advancing the range start past the bytes already on disk and
    appending. curl's own `-C -` cannot be combined with `-r` on every build (the
    Windows system curl rejects it), so it is not used.
    """
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have >= length:
        return
    end = start + length - 1
    with open(dest, "ab") as out:
        subprocess.run(
            ["curl", "-f", "-s", "-L", "--retry", "5", "--retry-delay", "3",
             "-r", f"{start + have}-{end}", url],
            stdout=out,
            check=True,
        )


def extract(url, name, meta, out_dir):
    header_offset, comp_size, size, comp_type = meta
    dest = os.path.join(out_dir, name)
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        print(f"  {name}: already complete")
        return

    # Local file header: 30 fixed bytes, then a variable name and extra field.
    header = dest + ".hdr"
    curl_range(url, header_offset, 30, header)
    with open(header, "rb") as fh:
        raw = fh.read()
    os.remove(header)
    if raw[:4] != b"PK\x03\x04":
        sys.exit(f"{name}: bad local header signature, archive layout changed")
    name_len, extra_len = struct.unpack("<HH", raw[26:30])
    data_offset = header_offset + 30 + name_len + extra_len

    staged = dest + ".part"
    print(f"  {name}: fetching {comp_size / 1e6:.1f} MB compressed")
    curl_range(url, data_offset, comp_size, staged)

    if comp_type == 0:
        os.replace(staged, dest)
    else:
        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
        with open(staged, "rb") as src, open(dest, "wb") as dst:
            while chunk := src.read(CHUNK):
                dst.write(decompressor.decompress(chunk))
            dst.write(decompressor.flush())
        os.remove(staged)

    actual = os.path.getsize(dest)
    if actual != size:
        sys.exit(f"{name}: got {actual} bytes, expected {size}")
    print(f"  {name}: wrote {actual / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", choices=sorted(ARCHIVES))
    parser.add_argument("members", nargs="*", help="basenames, e.g. 7.mp4 7.txt")
    parser.add_argument("--out", default="data/external/sard")
    parser.add_argument("--list", action="store_true", help="print contents and exit")
    args = parser.parse_args()

    url = ARCHIVES[args.archive]
    contents = index(url)

    if args.list or not args.members:
        for name, (_, comp, size, _) in sorted(
            contents.items(), key=lambda kv: -kv[1][2]
        ):
            print(f"{size / 1e6:10.1f} MB  ({comp / 1e6:7.1f} MB on wire)  {name}")
        return

    os.makedirs(args.out, exist_ok=True)
    missing = [m for m in args.members if m not in contents]
    if missing:
        sys.exit(f"not in archive: {', '.join(missing)}")

    total = sum(contents[m][1] for m in args.members)
    print(f"{len(args.members)} members, {total / 1e6:.1f} MB on wire")
    for member in args.members:
        extract(url, member, contents[member], args.out)


if __name__ == "__main__":
    main()
