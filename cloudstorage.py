import os
from supabase import create_client

def upload_image(local_file_path, object_path):
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    bucket = os.environ.get("SUPABASE_BUCKET", "wound-images")

    if not url or not key:
        raise RuntimeError("Set SUPABASE_URL and SUPABASE_KEY environment variables first.")

    client = create_client(url, key)
    with open(local_file_path, "rb") as f:
        response = client.storage.from_(bucket).upload(
            path=object_path,
            file=f,
            file_options={"content-type": "application/octet-stream", "upsert": "false"}
        )
    return response