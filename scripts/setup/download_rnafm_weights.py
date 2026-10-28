from multimolecule import RnaTokenizer

RnaTokenizer.from_pretrained("multimolecule/rnafm").save_pretrained("./rnafm_weights")
print("Tokenizer saved to ./rnafm_weights")

# Copy the model weights directly from HF cache
import shutil, pathlib, os

hf_cache = pathlib.Path.home() / ".cache/huggingface/hub/models--multimolecule--rnafm"
snapshots = sorted((hf_cache / "snapshots").iterdir())
if not snapshots:
    raise RuntimeError("No snapshots found in HF cache")

snapshot = snapshots[-1]
out = pathlib.Path("./rnafm_weights")
out.mkdir(exist_ok=True)

for f in snapshot.iterdir():
    dest = out / f.name
    if not dest.exists():
        shutil.copy2(f, dest)
        print(f"  Copied {f.name}")

print("Done — weights saved to ./rnafm_weights")
