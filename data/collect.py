import json
import time
import urllib.parse
from pathlib import Path

import requests
import tiktoken

UA = "my-deep-researcher-student/0.1 (Aiffel assignment)"
API = {"wv": "https://en.wikivoyage.org/w/api.php", "wp": "https://en.wikipedia.org/w/api.php"}

SEEDS = {
    "wv": [
        "Kansai", "Osaka", "Osaka/Kita", "Osaka/Minami", "Osaka/Bay Area",
        "Osaka/Tennoji", "Osaka/Osaka castle",
        "Kyoto", "Kyoto/Central", "Kyoto/Higashiyama", "Kyoto/Arashiyama",
        "Kyoto/North", "Kyoto/South", "Nara", "Kobe", "Uji", "Himeji",
        "Rail travel in Japan", "Kansai International Airport",
    ],
    "wp": [
        "ICOCA", "Osaka Metro", "Kyoto Municipal Transportation Bureau", "Keihan Main Line",
        "Hankyu Kyoto Main Line",
        "Fushimi Inari-taisha", "Kiyomizu-dera", "Kinkaku-ji", "Yasaka Shrine", "Osaka Castle",
        "Dōtonbori", "Universal Studios Japan", "Tōdai-ji",
        "Tourism in Japan", "Hanami",
    ],
}


def api(site, **params):
    params.update(format="json", formatversion=2)
    for wait in (5, 15, 30, 60, 120):
        r = requests.get(API[site], params=params, headers={"User-Agent": UA})
        if r.status_code != 429:
            time.sleep(1)
            return r.json()
        time.sleep(int(r.headers.get("Retry-After", wait)))
    r.raise_for_status()


def fetch_text(site, title):
    page = api(site, action="query", prop="extracts", explaintext=1, redirects=1, titles=title)["query"]["pages"][0]
    if "extract" not in page:
        raise SystemExit(f"본문 없음: {site}:{title} → {page}")
    return page["title"], page["extract"]


def fetch_links(site, title):
    targets, cont = set(), {}
    while True:
        data = api(site, action="query", generator="links", titles=title, gplnamespace=0,
                   gpllimit="max", redirects=1, **cont)
        targets |= {p["title"] for p in data.get("query", {}).get("pages", [])}
        if "continue" not in data:
            return targets
        cont = data["continue"]


docs = []
for site, titles in SEEDS.items():
    for t in titles:
        title, text = fetch_text(site, t)
        if any(d["id"] == f"{site}:{title}" for d in docs):
            print(f"넘겨주기 중복으로 건너뜀: {site}:{t} → {title}")
            continue
        docs.append({"id": f"{site}:{title}", "site": site, "title": title,
                     "url": API[site].replace("/w/api.php", "/wiki/") + urllib.parse.quote(title.replace(" ", "_")),
                     "text": text, "_out": fetch_links(site, title)})

ids = {(d["site"], d["title"]): d["id"] for d in docs}
links = {d["id"]: sorted(ids[(d["site"], t)] for t in d.pop("_out") if (d["site"], t) in ids and ids[(d["site"], t)] != d["id"])
         for d in docs}

Path(__file__).with_name("corpus.json").write_text(
    json.dumps({"docs": docs, "links": links}, ensure_ascii=False, indent=1), encoding="utf-8")

enc = tiktoken.encoding_for_model("gpt-4o-mini")
chars = sum(len(d["text"]) for d in docs)
tokens = sum(len(enc.encode(d["text"])) for d in docs)
inbound = {i: 0 for i in links}
for outs in links.values():
    for o in outs:
        inbound[o] += 1

print(f"{'id':40} {'chars':>8} {'out':>4} {'in':>4}")
for d in sorted(docs, key=lambda d: -len(d["text"])):
    print(f"{d['id']:40} {len(d['text']):>8,} {len(links[d['id']]):>4} {inbound[d['id']]:>4}")
print(f"\n문서 {len(docs)}건 · {chars:,}자 · {tokens:,} 토큰 (gpt-4o-mini 창 128,000 토큰의 {tokens / 128000:.2f}배)")
print("나가는 링크 0개:", [i for i, o in links.items() if not o])
print("들어오는 링크 0개:", [i for i, n in inbound.items() if not n])
