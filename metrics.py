import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
CITE = re.compile(r"«([^»]+)»")
ONLY_CITES = re.compile(r"^[\s(),.:근거]*(«[^»]+»[\s(),.:]*)+$")
LEAD_CITES = re.compile(r"^((?:«[^»]+»[\s,.]*)+)(.*)$")
SPLIT = re.compile(r"(?<=[.!?])\s+")
TITLE2ID = {d["title"]: d["id"] for d in json.loads((ROOT / "data/corpus.json").read_text(encoding="utf-8"))["docs"]}


def cite_ids(text):  # «문서 제목» → id (예전 기록처럼 id로 인용했으면 그대로)
    return [TITLE2ID.get(c, c) for c in CITE.findall(text)]


def sentences(text):
    """문장 분리 규칙 (바꾸지 않는다): 빈 줄·# 제목 줄 제외, 목록은 한 줄 = 한 문장,
    마침표 뒤 «id»는 앞 문장 몫, 인용만 있는 조각은 앞 문장에 합친다."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for s in SPLIT.split(line):
            lead = LEAD_CITES.match(s)
            if lead and out:  # 마침표 바로 뒤의 «id»는 앞 문장 몫, 나머지는 새 문장
                out[-1] += " " + lead.group(1).strip()
                s = lead.group(2)
            if ONLY_CITES.match(s) and out:
                out[-1] += " " + s
            elif s.strip():
                out.append(s)
    return out


def evaluate(st):
    sections = {}
    cites, false_cites = [], []
    for i, d in sorted(st["drafts"].items(), key=lambda x: int(x[0])):
        sents = sentences(d["text"])
        cited = sum(bool(CITE.search(s)) for s in sents)
        c = cite_ids(d["text"])
        cites += c
        false_cites += [x for x in c if x not in d["read"]]
        sections[int(i) + 1] = {"문장": len(sents), "근거 문장": cited,
                                "근거 비율": round(cited / len(sents), 3) if sents else 0.0}

    n_sent = sum(s["문장"] for s in sections.values())
    n_cited = sum(s["근거 문장"] for s in sections.values())
    reads = [doc for r in st["results"] for doc in r["new_read"]]
    used = set(cites)
    top = Counter(cites).most_common(1)
    return {
        # 신호 — 높낮이를 견준다
        "근거 비율": round(n_cited / n_sent, 3) if n_sent else 0.0,
        "읽고 안 쓴 문서": sorted(set(reads) - used),
        "편중": round(top[0][1] / len(cites), 3) if cites else 0.0,
        "중복률": round(1 - len(set(reads)) / len(reads), 3) if reads else 0.0,
        # 경보 — 0이어야 한다
        "허위 인용": false_cites,
        "반복 문장": sum(r.get("repeats", 0) for r in st["results"]),  # 코드가 지운 되풀이 문장 수 (기각된 바퀴 포함)
        # 기록 — 지표가 아니다
        "절별": sections,
        "코디네이터가 본 글자": st["coordinator_chars"],
        "리서처가 본 글자(합)": sum(r["researcher_chars"] for r in st["results"]),
        "리서처 한 명이 본 글자(최대)": max(r["researcher_chars"] for r in st["results"]),
        "보고서 글자": len(st["report"]),
        "다시 나간 절": sum(r["round"] > 1 for r in st["results"]),
    }


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    runs = [json.loads(l) for l in (ROOT / "output/runs.jsonl").read_text(encoding="utf-8").splitlines()][-n:]
    for st in runs:
        m = evaluate(st)
        alarm = "⚠️ " if m["허위 인용"] or m["반복 문장"] else "✅ "
        print(f"{alarm}{st['qid']} {st['stamp']} | 근거 {m['근거 비율']:.1%} | 읽고 안 씀 {len(m['읽고 안 쓴 문서'])}건 | "
              f"편중 {m['편중']:.1%} | 중복 {m['중복률']:.1%} | 허위 인용 {len(m['허위 인용'])} {m['허위 인용'] or ''}| "
              f"반복 문장 {m['반복 문장']}")
        print(f"   절별 근거 {[f'{s['근거 비율']:.0%}' for s in m['절별'].values()]} | 코디네이터 {m['코디네이터가 본 글자']:,}자 · "
              f"리서처 합 {m['리서처가 본 글자(합)']:,}자 (한 명 최대 {m['리서처 한 명이 본 글자(최대)']:,}자) | "
              f"보고서 {m['보고서 글자']:,}자 | 다시 나간 절 {m['다시 나간 절']}")
