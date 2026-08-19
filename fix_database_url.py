from pathlib import Path

SECRETS_FILE = Path(".streamlit") / "secrets.toml"

# Put your CURRENT Supabase DATABASE_URL here.
# Do NOT post the password in chat.
NEW_DATABASE_URL = "postgresql://YOUR_SUPABASE_USERNAME:YOUR_PASSWORD@aws-0-ap-south-1.pooler.supabase.com:5432/postgres"

if not SECRETS_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {SECRETS_FILE.resolve()}"
    )

lines = SECRETS_FILE.read_text(encoding="utf-8").splitlines()

new_lines = []
database_url_found = False

for line in lines:
    if line.strip().startswith("DATABASE_URL"):
        new_lines.append(
            f'DATABASE_URL = "{NEW_DATABASE_URL}"'
        )
        database_url_found = True
    else:
        new_lines.append(line)

if not database_url_found:
    new_lines.append(
        f'DATABASE_URL = "{NEW_DATABASE_URL}"'
    )

SECRETS_FILE.write_text(
    "\n".join(new_lines) + "\n",
    encoding="utf-8",
)

print("DATABASE_URL updated successfully.")
print("File:", SECRETS_FILE.resolve())

# Safe verification — password is NOT printed.
saved = SECRETS_FILE.read_text(encoding="utf-8")

for line in saved.splitlines():
    if line.strip().startswith("DATABASE_URL"):
        print("Saved DATABASE_URL host:")
        print(line.split("@")[-1])
        break