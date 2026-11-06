import json
import statistics as stats
import sys

import baseline
import graph
from metrics import evaluate

ROOT = graph.ROOT
QIDS = ["Q3", "Q6", "Q8"]  # 흩어짐 · 도시 비교 · 한 대상
REPEATS = 3                # 같은 설정도 시행마다 흔들리므로 3번씩
QS = {q["id"]: q["question"] for q in json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8"))}


def summary(st):
    m = evaluate(st)
    return {"근거 비율": m["근거 비율"], "읽고 안 쓴 문서": len(m["읽고 안 쓴 문서"]), "편중": m["편중"],
            "중복률": m["중복률"], "허위 인용": len(m["허위 인용"]), "반복 문장": m["반복 문장"],
            "발췌 불일치": sum(m["발췌 불일치"].values()), "보고서 글자": m["보고서 글자"],
            "읽기 횟수": sum(len(r["new_read"]) for r in st["results"])}


def switch_check(st, mode):
    """스위치가 로그만 찍고 코드 경로는 그대로인 사고를 막는다 — 실제 기록에서 확인"""
    starts = [s["start_doc"] for s in st["sections"]]
    if mode == "배정 끔":
        assert all(s is None for s in starts), f"배정 끔인데 시작 문서가 있음: {starts}"
        first = [r["new_read"][0] for r in st["results"] if r["round"] == 1 and r["new_read"]]
        return f"시작 문서 없음 → 리서처가 고른 첫 문서 {first}"
    if mode == "기본":
        assert all(starts), f"기본인데 시작 문서가 비었음: {starts}"
        return f"배정된 시작 문서 {starts}"
    return f"읽기 예산 {st['budget']}건 → 실제 {len(st['drafts']['0']['read'])}건"


if __name__ == "__main__":
    rows = []
    for qid in QIDS:
        for rep in range(1, REPEATS + 1):
            base = graph.run(QS[qid])
            budget = sum(len(r["new_read"]) for r in base["results"])  # 대조군 예산 = 방금 기본이 실제로 읽은 횟수
            runs = [("기본", base), ("단일", baseline.run(QS[qid], budget)), ("배정 끔", graph.run(QS[qid], assign=False))]
            for mode, st in runs:
                stamp = graph.save(qid, st, mode)
                note = switch_check(st, mode)
                rows.append({"qid": qid, "rep": rep, "mode": mode, "stamp": stamp, "check": note, **summary(st)})
                print(f"{qid} #{rep} {mode:5} | {note}", flush=True)

    keys = ["근거 비율", "읽고 안 쓴 문서", "편중", "중복률", "허위 인용", "반복 문장", "발췌 불일치", "보고서 글자", "읽기 횟수"]
    table = {}
    for qid in QIDS:
        for mode in ["기본", "배정 끔", "단일"]:
            rs = [r for r in rows if r["qid"] == qid and r["mode"] == mode]
            table[f"{qid} {mode}"] = {k: {"평균": round(stats.mean(r[k] for r in rs), 3),
                                          "최소": min(r[k] for r in rs), "최대": max(r[k] for r in rs)} for k in keys}
    (ROOT / "output/ablation.json").write_text(json.dumps({"rows": rows, "table": table}, ensure_ascii=False, indent=1),
                                               encoding="utf-8")

    print(f"\n{'':10}" + "".join(f"{k:>14}" for k in keys))
    for name, t in table.items():
        cells = []
        for k in keys:
            lo, hi = t[k]["최소"], t[k]["최대"]
            pct = k in ("근거 비율", "편중", "중복률")
            cells.append(f"{lo:.0%}~{hi:.0%}" if pct else f"{lo}~{hi}")
        print(f"{name:10}" + "".join(f"{c:>14}" for c in cells))
    sys.exit(0)
