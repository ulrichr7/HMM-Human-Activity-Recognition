from pathlib import Path
import argparse, io, zipfile, re
import pandas as pd

ACTIVITIES = ["standing", "walking", "jumping", "still"]

def import_archive(zip_path, activity, output_root):
    zip_path = Path(zip_path); output_root = Path(output_root)
    with zipfile.ZipFile(zip_path) as outer:
        nested = [n for n in outer.namelist() if n.lower().endswith(".zip") and not n.startswith("__MACOSX/")]
        def idx(n):
            m = re.search(rf"{activity}_(\d+)", Path(n).name, re.I)
            return int(m.group(1)) if m else 999
        for n in sorted(nested, key=lambda n: (idx(n), n)):
            rid = f"{activity}_{idx(n):02d}"
            out = output_root / "data" / "raw" / activity / rid
            out.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(outer.read(n))) as inner:
                for f in inner.namelist():
                    if f.startswith("__MACOSX/") or f.endswith("/") or not f.lower().endswith(".csv"):
                        continue
                    name = Path(f).name
                    b = inner.read(f)
                    if name.lower() == "metadata.csv" and b.strip():
                        df = pd.read_csv(io.BytesIO(b), dtype=str)
                        df = df.drop(columns=[c for c in df.columns if c.strip().lower() == "device id"], errors="ignore")
                        b = df.to_csv(index=False).encode()
                    (out/name).write_bytes(b)
            (out/"SOURCE_FILE.txt").write_text(f"Source archive: {zip_path.name}\nNested archive: {n}\nActivity: {activity}\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Import one Sensor Logger outer ZIP into data/raw, sanitizing device UUID metadata.")
    p.add_argument("activity", choices=ACTIVITIES)
    p.add_argument("zip_path")
    p.add_argument("--project-root", default=".")
    a = p.parse_args(); import_archive(a.zip_path, a.activity, a.project_root)
