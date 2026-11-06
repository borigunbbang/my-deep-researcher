import json
import sys
import time

import tiktoken
from langchain_openai import ChatOpenAI

from graph import CFG, DOCS, LINE_RULES, LINKS, LLM, ROOT, Pick, card, check_lines, render

# 같은 자료(corpus) · 같은 모델 · 같은 도구(카드 보고 다음 문서 고르기, 링크 따라가기) · 같은 읽기 예산
# 출력 상한은 멀티의 리서처 상한(4,000) × 절 상한(4) — 대조군만 짧게 묶지 않는다.
# 단, 창(128K)에서 프롬프트를 뺀 만큼을 넘지 않게 줄인다 (넘기면 API가 거절한다)
MAX_OUT = 4000 * CFG["max_sections"]
ENC = tiktoken.encoding_for_model(CFG["model"])

PICK_PROMPT = """너는 여행 리서처다. 질문에 답할 보고서를 혼자 쓴다.
질문: {question}
지금까지 읽은 문서: {read}
다음에 읽을 문서를 후보 중에서 하나 고른다.

후보 카드
{cards}"""

WRITE_PROMPT = """너는 여행 리서처다. 아래 문서를 읽고 질문에 답하는 보고서를 혼자 쓴다.
질문: {question}

읽은 문서
{docs}

규칙 (문서를 다 읽은 뒤 지킨다)
- 보고서를 최대 {max_sections}개 절로 나누고 절마다 '## 절 제목' 줄로 시작한다.
- 위 문서에 적힌 내용만 쓴다. 문서에 없는 것은 쓰지 않는다. 근거가 있는 만큼 자세히 쓴다.
{line_rules}"""


def multi_budget(qid):
    """같은 질문을 멀티 에이전트가 가장 최근에 돌렸을 때 실제로 읽은 횟수"""
    runs = [json.loads(l) for l in (ROOT / "output/runs.jsonl").read_text(encoding="utf-8").splitlines()]
    multi = [r for r in runs if r["qid"] == qid and r.get("mode", "기본") == "기본"]
    return sum(len(r["new_read"]) for r in multi[-1]["results"]), multi[-1]["stamp"]


def run(question, budget):
    read, seen = [], 0
    while len(read) < budget:
        frontier = {l for d in read for l in LINKS[d]}
        cands = [d for d in DOCS if d in frontier and d not in read] or [d for d in DOCS if d not in read]
        if not cands:
            break
        prompt = PICK_PROMPT.format(question=question, read=", ".join(read) or "없음",
                                    cards="\n".join(card(d) for d in cands))
        seen += len(prompt)
        pick = LLM.with_structured_output(Pick).invoke(prompt).doc_id
        read.append(pick if pick in cands else cands[0])

    docs = "\n\n".join(f"### {DOCS[d]['title']}\n{DOCS[d]['text']}" for d in read)
    prompt = WRITE_PROMPT.format(question=question, max_sections=CFG["max_sections"], line_rules=LINE_RULES, docs=docs)
    seen += len(prompt)
    tokens = len(ENC.encode(prompt))
    max_out = min(MAX_OUT, CFG["context_window"] - tokens - 1000)
    raw = ChatOpenAI(model=CFG["model"], temperature=0, max_tokens=max_out).invoke(prompt).content

    lines, repeats, dropped = check_lines(raw, set(read))  # 멀티와 같은 대조 코드
    text = render(lines)
    return {"question": question, "mode": "단일", "budget": budget, "prompt_tokens": tokens, "max_out": max_out,
            "sections": [{"title": "단일 에이전트", "role": "-", "start_doc": read[0]}],
            "drafts": {"0": {"text": text, "lines": lines, "read": read, "round": 1, "status": "-", "reason": "-"}},
            "results": [{"idx": 0, "round": 1, "new_read": read, "researcher_chars": seen,
                         "repeats": repeats, "dropped": dropped}],
            "report": f"# {question}\n\n{text}",
            "coordinator_chars": 0, "plan_fixes": [], "decisions": []}


if __name__ == "__main__":
    qs = {q["id"]: q["question"] for q in json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8"))}
    qid = sys.argv[1]
    budget, ref = multi_budget(qid)
    st = run(qs[qid], budget)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    (ROOT / "output/reports" / f"{qid}_{stamp}_baseline.md").write_text(st["report"], encoding="utf-8")
    with open(ROOT / "output/runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"qid": qid, "stamp": stamp, "ref": ref, **st}, ensure_ascii=False) + "\n")

    used = len(st["drafts"]["0"]["read"])
    print(f"읽기 예산 {budget}건 (멀티 {ref} 기준) → 실제로 읽음 {used}건 {'✅ 다 씀' if used == budget else '⚠️ 덜 씀'}")
    print(f"읽은 문서 {st['drafts']['0']['read']}")
    print(f"쓰기 프롬프트 {st['prompt_tokens']:,} 토큰 (창 {CFG['context_window']:,}의 {st['prompt_tokens'] / CFG['context_window']:.0%}) · 출력 상한 {st['max_out']:,} 토큰")
    print(f"보고서 {len(st['report']):,}자 → output/reports/{qid}_{stamp}_baseline.md")
