import requests, tarfile, tempfile

url = "https://ndownloader.figshare.com/files/11767817"
resp = requests.get(url, stream=True)

with tempfile.NamedTemporaryFile(suffix=".tar.bz2") as tmp:
    for chunk in resp.iter_content(chunk_size=8192):
        tmp.write(chunk)
        if tmp.tell() > 2000000: # 2MB is enough to get the header and some files
            break
    tmp.flush()
    try:
        with tarfile.open(tmp.name, "r:bz2") as tar:
            for i, member in enumerate(tar.getmembers()):
                print(member.name)
                if i > 20: break
    except Exception as e:
        print(e)
