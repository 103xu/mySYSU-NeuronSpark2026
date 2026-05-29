import zipfile, os

OUTPUT = r"C:\Users\32010\ai-competition\04"
zf = zipfile.ZipFile(f"{OUTPUT}/NS-2026-04-answer.zip", "w", zipfile.ZIP_DEFLATED)
zf.write(f"{OUTPUT}/results.csv", "results.csv")
zf.close()
size = os.path.getsize(f"{OUTPUT}/NS-2026-04-answer.zip")
print(f"NS-2026-04-answer.zip created: {size} bytes")

# Verify
with zipfile.ZipFile(f"{OUTPUT}/NS-2026-04-answer.zip", "r") as zf2:
    print(f"Contents: {zf2.namelist()}")
