import os, pathlib, sys, time
sys.path.insert(0,"/root/reelkit"); import drive
ENV={"LILBEE_DATA":"/root/reel-data","LILBEE_WIKI":"true","LILBEE_CHAT_MODEL":os.environ["LILBEE_CHAT_MODEL"],
     "LILBEE_EMBEDDING_MODEL":os.environ["LILBEE_EMBEDDING_MODEL"],"VIRTUAL_ENV":"/root/lilbee/.venv"}
def rp(s): return [ln.split("│",1)[1] for ln in s.screen().splitlines() if "│" in ln]
def bc(s):
    for ln in s.screen().splitlines():
        if "entities >" in ln: return ln.strip()
    return "?"
s=drive.Session("p4win",128,44,pathlib.Path("/root/out/probe4.cast")); s.start("/root/lilbee/.venv/bin/lilbee",env=ENV)
try:
    s.wait_for(r"personal encyclopedia|Welcome to lilbee",150); time.sleep(1.2)
    if "Welcome" in s.screen(): s.esc(2); time.sleep(1.6)
    else: s.esc(2); time.sleep(0.6)
    for _ in range(5): s.key("]",after=0.6)
    s.wait_for(r"Filter pages",90); time.sleep(1.2)
    s.wait_for(r"warming up", absent=True, timeout=200); time.sleep(1.0); print("WARM done")
    # Earth tree + scroll
    s.key("g",after=0.4); s.key(*(["j"]*15),after=0.04); s.key("enter",after=0.9); time.sleep(1.4)
    print("EARTH:", bc(s))
    s.key("Tab",after=1.5); s.key(*(["Down"]*34),after=0.045); time.sleep(1.0)
    print("EARTH scrolled body:", [l for l in rp(s) if l.strip()][3:7])
    # Neptune filter
    s.key("/",after=0.4); s.key("C-u",after=0.2); s.key(*"neptune",after=0.11); time.sleep(1.0)
    s.key("Tab",after=0.4); s.key("G",after=0.5); s.key("enter",after=0.9); time.sleep(1.4)
    print("NEPTUNE:", bc(s))
    # Mercury filter
    s.key("/",after=0.4); s.key("C-u",after=0.2); s.key(*"mercury",after=0.11); time.sleep(1.0)
    s.key("Tab",after=0.4); s.key("G",after=0.5); s.key("enter",after=0.9); time.sleep(1.4)
    print("MERCURY:", bc(s))
finally: s.kill()
