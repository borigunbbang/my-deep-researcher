import difflib
import json
import operator
import re
import sys
import time
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

load_dotenv()
ROOT = Path(__file__).parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
CORPUS = json.loads((ROOT / "data/corpus.json").read_text(encoding="utf-8"))
DOCS = {d["id"]: d for d in CORPUS["docs"]}
LINKS = CORPUS["links"]
LLM = ChatOpenAI(model=CFG["model"], temperature=0)
CITE = re.compile(r"«([^»]+)»")


def card(doc_id):
    return f"- {doc_id} : {DOCS[doc_id]['text'][:CFG['card_chars']]}".replace("\n", " ")


TITLE2ID = {d["title"]: d["id"] for d in CORPUS["docs"]}


def valid_cites(text, read):  # 인용은 «문서 제목» — id로 바꿔 읽은 문서인지 본다
    return [c for c in CITE.findall(text) if TITLE2ID.get(c) in read]


# ---------- 발췌 대조: 문장마다 붙인 영어 원문 구절이 그 문서에 글자 그대로 있는지 코드가 본다 ----------
# 문장 «제목» ⟦구절⟧ 한 묶음. 한 줄에 여러 묶음이 이어져도 하나씩 떼어 따로 대조한다
# (구절에 ⟦⟧를 못 넣게 해야 첫 ⟦부터 마지막 ⟧까지를 한 구절로 삼키지 않는다)
SEG = re.compile(r"(?P<s>[^«»⟦⟧]+?)\s*«(?P<t>[^«»]+)»\s*⟦(?P<q>[^⟦⟧]+)⟧[\s.。]*")
MIN_QUOTE = 30  # 구절이 원문과 연속 30자 이상 그대로 겹쳐야 통과 (Q3 기록에서 지어낸 구절은 18자 이하, 옮긴 구절은 44자 이상으로 갈렸다)


def norm(x):
    x = x.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", x).strip(" .,;:'\"")


NORM = {i: norm(d["text"]) for i, d in DOCS.items()}


def overlap(q, doc):
    """구절과 원문이 연속으로 그대로 겹치는 가장 긴 길이 (한 글자씩 바꿔 옮긴 구절은 살리고, 지어낸 구절은 걸러낸다)"""
    if q in doc:
        return len(q)
    return difflib.SequenceMatcher(None, q, doc, autojunk=False).find_longest_match(0, len(q), 0, len(doc)).size


def check_lines(raw, read):
    """원고를 '문장 «제목» ⟦구절⟧' 묶음 단위로 대조해 통과한 것만 남긴다.
    반환: 남은 문장, 되풀이 수, 지운 것(이유 포함), 제목을 바로잡은 것"""
    kept, seen, repeats, dropped, fixed = [], set(), 0, [], []
    for line in (l.strip() for l in raw.splitlines()):
        if not line:
            continue
        if line.startswith("#"):
            kept.append({"heading": line})
            continue
        segs = list(SEG.finditer(line))
        rest = SEG.sub("", line).strip(" .。")
        if rest:  # 묶음 밖에 남은 글 = 인용·구절 없이 쓴 문장
            dropped.append({"line": rest, "why": "형식 (인용·발췌 없음)"})
        for m in segs:
            unit = m.group(0).strip()
            if unit in seen:  # 같은 문장이 되풀이되면 두 번째부터 지운다 (폭주 대비)
                repeats += 1
                continue
            seen.add(unit)
            title, q = m["t"].strip(), norm(m["q"])
            doc = TITLE2ID.get(title)
            if len(q) < MIN_QUOTE:
                dropped.append({"line": unit, "why": "발췌가 너무 짧음"})
                continue
            if doc not in read or overlap(q, NORM[doc]) < MIN_QUOTE:
                # 제목 보정: 구절이 실제로 들어 있는 읽은 문서가 있으면 그 문서로 제목을 바로잡는다 (구절이 출처를 증명한다)
                best = max(((overlap(q, NORM[d]), d) for d in read if d != doc), default=(0, None))
                if best[0] < MIN_QUOTE:
                    dropped.append({"line": unit, "why": "읽지 않은 문서" if doc not in read else "발췌가 문서에 없음"})
                    continue
                fixed.append({"line": unit, "from": title, "to": DOCS[best[1]]["title"]})
                title = DOCS[best[1]]["title"]
            kept.append({"s": m["s"].strip(), "title": title, "quote": m["q"].strip()})
    return kept, repeats, dropped, fixed


def render(lines, with_quote=False):
    """대조를 통과한 줄을 문단으로 잇는다 (# 줄은 절 제목)"""
    out, body = [], []
    for l in lines + [{"heading": None}]:
        if "heading" in l:
            if body:
                out.append(" ".join(body))
                body = []
            if l["heading"]:
                out.append(l["heading"])
        else:
            body.append(f"{l['s']} «{l['title']}»" + (f" ⟦{l['quote']}⟧" if with_quote else ""))
    return "\n\n".join(out)


# ---------- 상태 ----------
class State(TypedDict, total=False):
    question: str
    sections: list[dict]          # 절마다 title·role·start_doc
    plan_fixes: list[str]
    coordinator_chars: int        # 코디네이터가 본 글자 수 (카드 + 올라온 원고)
    round: int
    todo: list[int]               # 이번 바퀴에 내보낼 절 번호
    results: Annotated[list[dict], operator.add]   # 리서처가 올린 것 (바퀴마다 쌓임)
    drafts: dict                  # 절 번호 → 채택된 원고
    decisions: list[str]          # 두 번째 원고 채택/기각 기록
    report: str
    assign: bool                  # 스위치: 시작 문서 배정 (ablation에서 끈다)


# ---------- ① 기획 ----------
class Section(BaseModel):
    title: str = Field(description="절 제목 (한국어, 장소나 주제 덩어리)")
    role: str = Field(description="역할 명단 중 하나")
    start_doc: str = Field(description="가장 먼저 읽을 문서 id (카드에 적힌 id 그대로)")


class Plan(BaseModel):
    sections: list[Section]


PLAN_PROMPT = """너는 여행 리서치 팀의 코디네이터다. 질문에 답할 보고서의 목차를 짠다.

규칙
- 절은 최대 {max_sections}개. 질문이 좁으면 절을 줄인다. 문서 한 건으로 답이 나오는 질문이면 1절만 짠다.
- 절은 내용으로 나눈다: 장소(지역·명소) 또는 주제 덩어리(이동·시기·규정). '개요·일정표·분석·결론'처럼 형식으로 나누지 않는다.
- 질문에 나온 장소는 빠짐없이 어느 절엔가 들어가야 한다.
- 절마다 읽을 문서가 서로 갈리도록 나눈다.
- 절마다 역할 명단에서 하나를 고른다. 역할은 관점일 뿐이니 절 제목을 역할 이름(관광지·동선·교통·시기 등)으로 짓지 않는다.
- 절마다 시작 문서를 아래 카드의 id 그대로 하나 고른다. 절끼리 시작 문서가 겹치면 안 된다.

예시 (다른 도시 질문: "도쿄 5일 여행을 관광지·교통·시기를 고려해 짜면?")
나쁜 목차 — 질문의 항목을 그대로 절로 옮겨서 모든 절이 같은 도시 문서를 읽는다
  1. 관광지 개요  2. 교통수단  3. 이동 동선  4. 여행 시기
좋은 목차 — 장소로 나눠서 절마다 읽을 문서가 갈린다. 교통·시기는 각 절 안에서 다루거나, 전용 문서가 따로 있을 때만 절로 뺀다
  1. 시부야·신주쿠  2. 아사쿠사·우에노  3. 근교 닛코 당일치기  4. 공항과 도시 사이 이동

역할 명단
{roles}

문서 카드 (id : 앞부분)
{cards}

질문: {question}"""


def plan(state: State) -> State:
    roles = "\n".join(f"- {k}: {v}" for k, v in CFG["roles"].items())
    cards = "\n".join(card(i) for i in DOCS)
    prompt = PLAN_PROMPT.format(max_sections=CFG["max_sections"], roles=roles, cards=cards, question=state["question"])
    out = LLM.with_structured_output(Plan).invoke(prompt)

    fixes, sections, used = [], [], set()
    for s in out.sections[:CFG["max_sections"]]:
        role, doc = s.role, s.start_doc
        if role not in CFG["roles"]:
            fixes.append(f"역할 '{role}' → {CFG['default_role']}")
            role = CFG["default_role"]
        if doc in used:  # 시작 문서가 겹치면 나눌 필요가 없다는 신호 — 그 절은 버린다
            fixes.append(f"절 '{s.title}' 버림 (시작 문서 {doc} 중복)")
            continue
        if doc not in DOCS:  # 지어낸 id는 가장 가까운 실제 id로 바꾼다
            free = [i for i in DOCS if i not in used]
            near = difflib.get_close_matches(doc, free, n=1, cutoff=0)
            fixes.append(f"시작 문서 '{doc}' → {near[0]}")
            doc = near[0]
        used.add(doc)
        sections.append({"title": s.title, "role": role, "start_doc": doc})
    if len(out.sections) > CFG["max_sections"]:
        fixes.append(f"절 {len(out.sections)}개 → {CFG['max_sections']}개로 자름")
    if not sections:
        sections = [{"title": "개요", "role": CFG["default_role"], "start_doc": next(iter(DOCS))}]
        fixes.append("목차가 비어 '개요' 한 절로 채움")
    if not state.get("assign", True):  # 배정 끔: 목차·역할은 그대로, 시작 문서만 지운다 (리서처가 전체 카드에서 스스로 고른다)
        for sec in sections:
            sec["start_doc"] = None
    return {"sections": sections, "plan_fixes": fixes, "coordinator_chars": len(prompt),
            "round": 1, "todo": list(range(len(sections))), "drafts": {}, "decisions": []}


# ---------- ② 배치 ----------
def dispatch(state: State):
    secs, drafts = state["sections"], state["drafts"]
    sends = []
    for i in state["todo"]:
        avoid = {s["start_doc"] for j, s in enumerate(secs) if j != i and s["start_doc"]}
        if state["round"] > 1:  # 2바퀴: 다른 절이 실제로 읽은 문서 + 자기가 이미 읽은 문서까지 피한다
            avoid |= {d for j, dr in drafts.items() if j != i for d in dr["read"]}
            avoid |= set(drafts[i]["read"])
        others = [f"- {s['title']} (시작 문서 {s['start_doc'] or '없음'})" for j, s in enumerate(secs) if j != i]
        sends.append(Send("research", {
            "question": state["question"], "idx": i, "round": state["round"], "section": secs[i],
            "others": others, "avoid": sorted(avoid), "prev": drafts.get(i)}))
    return sends


# ---------- ③ 조사 (서브에이전트) ----------
class Pick(BaseModel):
    doc_id: str = Field(description="다음에 읽을 문서 id (후보 카드의 id 그대로)")


WRITER = ChatOpenAI(model=CFG["model"], temperature=0, max_tokens=4000)  # 폭주해도 그래프가 멈추지 않게
REPORT_LINE = re.compile(r"^\s*신고\s*:\s*(충분|부족)\s*\|?\s*(.*)$", re.M)


PICK_PROMPT = """너는 {role}이다. {role_desc}
질문: {question}
네가 맡은 절: {title}
다음에 읽을 문서를 후보 중에서 하나 고른다.

후보 카드
{cards}"""

LINE_RULES = f"""- 한국어로, 한 줄에 한 문장씩 쓴다. 목록 기호·표를 쓰지 않는다.
- 읽은 문서마다 맡은 범위(질문)에 맞는 내용을 찾아 문서 하나당 {CFG["min_lines_per_doc"]}문장 이상 쓴다. 맞는 내용이 없는 문서는 건너뛴다.
- 줄마다 끝에 그 문장의 근거 문서 제목을 «» 안에, 이어서 근거가 된 영어 원문 구절을 ⟦⟧ 안에 붙인다.
- 제목은 '### ' 뒤에 적힌 것을 글자 그대로 쓴다. 문서 안의 소제목(== ... ==)은 문서 제목이 아니다.
- 원문 구절은 그 문서에서 이어진 한 구절을 30자 이상 글자 그대로 복사한다. 바꿔 쓰거나 줄임표(...)로 잇지 않는다.
- 원문 구절을 댈 수 없는 문장은 쓰지 않는다. 맺음말·요약 문장도 쓰지 않는다. 코드가 원문과 대조해 맞지 않는 줄은 지운다.

형식 예시 (다른 도시, 문서 제목과 구절은 예시일 뿐이다)
시부야 스크램블 교차로는 한 번에 수천 명이 건너는 곳이다. «Tokyo/Shibuya» ⟦thousands of people cross at once⟧
야마노테선을 타면 신주쿠까지 7분이 걸린다. «Tokyo/Shibuya» ⟦the Yamanote Line takes 7 minutes to Shinjuku⟧"""

WRITE_PROMPT = """너는 {role}이다. {role_desc}
질문: {question}
네가 맡은 절: {title}

다른 리서처가 맡은 절 (그 내용은 쓰지 않는다)
{others}

이 절에 이미 쓴 문장 (되풀이하지 않는다)
{written}

지금 읽을 문서
{docs}

규칙 (문서를 다 읽은 뒤 지킨다)
- 질문 전체에 답하지 않는다. 네가 맡은 절의 범위만 쓴다.
- 위 문서에 적힌 내용만 쓴다. 문서에 없는 것은 쓰지 않는다. 근거가 있는 만큼 자세히 쓴다. 소제목을 쓰지 않는다.
{line_rules}"""

REPORT_PROMPT = """너는 {role}이다.
네가 맡은 절: {title}
아래는 네가 문서 {n}건을 읽고 쓴 절 원고다 (문장 {lines}개). 이 절 원고의 근거가 충분했는지 신고한다.

판단 기준
- 기본은 '충분'이다.
- 다음 둘 중 하나에 해당할 때만 '부족'이다.
  ① 원고 문장이 {min_lines}개 미만이다.
  ② 절 제목에 나온 장소나 주제를 직접 다룬 문장이 하나도 없다.
- 다른 리서처가 맡은 절(아래)의 내용이 없는 것은 부족이 아니다.
{others}

원고
{draft}

아래 형식으로 한 줄만 적는다. 이유에는 ①·② 중 무엇에 해당하는지(또는 해당하지 않는지) 적는다.
신고: 충분 | 이유
신고: 부족 | 이유"""


def research(task: dict) -> State:
    sec, avoid = task["section"], set(task["avoid"])
    role_line = {"role": sec["role"], "role_desc": CFG["roles"][sec["role"]],
                 "question": task["question"], "title": sec["title"]}
    seen = 0
    read = [] if task["prev"] or not sec["start_doc"] else [sec["start_doc"]]
    pool = task["prev"]["read"] if task["prev"] else read
    while len(read) < CFG["docs_per_section"]:
        frontier = {l for d in pool + read for l in LINKS[d]}
        cands = [d for d in DOCS if d in frontier and d not in read and d not in avoid]
        if not cands:  # 링크가 막히면 남은 문서 전체로 넓힌다
            cands = [d for d in DOCS if d not in read and d not in avoid and d not in pool]
        if not cands:
            break
        prompt = PICK_PROMPT.format(**role_line, cards="\n".join(card(d) for d in cands))
        seen += len(prompt)
        pick = LLM.with_structured_output(Pick).invoke(prompt).doc_id
        read.append(pick if pick in cands else cands[0])

    # 쓰기는 문서 한 건씩 따로 부른다 — 한 번에 주면 맨 앞 문서에서만 쓰고 나머지를 버렸다
    # 2바퀴는 지난 원고 줄(이미 대조 통과)을 그대로 이어받고 새 문서만 쓴다 — 제목을 갈아 끼울 일이 없다
    all_read = (task["prev"]["read"] if task["prev"] else []) + read
    lines = [l for l in task["prev"]["lines"] if "s" in l] if task["prev"] else []
    repeats, dropped, fixed = 0, [], []
    for d in read:
        written = "\n".join(f"{l['s']} «{l['title']}»" for l in lines) or "- 없음"
        prompt = WRITE_PROMPT.format(**role_line, others="\n".join(task["others"]) or "- 없음", written=written,
                                     docs=f"### {DOCS[d]['title']}\n{DOCS[d]['text']}", line_rules=LINE_RULES)
        seen += len(prompt)
        got, rep, drop, fix = check_lines(WRITER.invoke(prompt).content, set(all_read))
        known = {l["s"] for l in lines}
        repeats += rep + sum(l["s"] in known for l in got if "s" in l)
        lines += [l for l in got if "s" in l and l["s"] not in known]
        dropped += drop
        fixed += fix

    draft = render(lines)
    prompt = REPORT_PROMPT.format(role=sec["role"], title=sec["title"], n=len(all_read), draft=draft or "(비어 있음)",
                                  lines=len(lines), min_lines=CFG["min_lines_per_doc"],
                                  others="\n".join(task["others"]) or "- 없음")
    seen += len(prompt)
    m = REPORT_LINE.search(LLM.invoke(prompt).content)
    status, reason = (m.group(1), m.group(2).strip()) if m else ("부족", "신고 줄 없음")
    return {"results": [{"idx": task["idx"], "round": task["round"], "text": draft, "lines": lines,
                         "status": status, "reason": reason, "read": all_read, "new_read": read,
                         "researcher_chars": seen, "repeats": repeats, "dropped": dropped, "fixed": fixed}]}


# ---------- ④ 점검 ----------
def check(state: State) -> State:
    drafts, decisions = dict(state["drafts"]), list(state["decisions"])
    for r in [r for r in state["results"] if r["round"] == state["round"]]:
        old = drafts.get(r["idx"])
        # 두 번째 원고는 유효하게 인용한 '서로 다른 문서' 수가 줄지 않을 때만 채택한다
        # (인용 횟수로 견주면 같은 문장을 되풀이한 원고가 이긴다)
        if old and len(set(valid_cites(r["text"], r["read"]))) < len(set(valid_cites(old["text"], old["read"]))):
            decisions.append(f"절{r['idx'] + 1} {r['round']}바퀴 원고 기각 (인용한 문서 종류 감소)")
            continue
        if old:
            decisions.append(f"절{r['idx'] + 1} {r['round']}바퀴 원고 채택")
        drafts[r["idx"]] = r
    seen = state["coordinator_chars"] + sum(len(r["text"]) for r in state["results"] if r["round"] == state["round"])
    todo = [i for i, d in drafts.items() if d["status"] == "부족" and d["round"] == state["round"]]
    if state["round"] >= CFG["max_rounds"]:
        todo = []
    return {"drafts": drafts, "decisions": decisions, "todo": todo,
            "round": state["round"] + 1 if todo else state["round"], "coordinator_chars": seen}


def after_check(state: State):
    return dispatch(state) if state["todo"] else "synthesize"


# ---------- ⑤ 종합 ----------
def synthesize(state: State) -> State:
    parts = [f"# {state['question']}\n"]
    for i, s in enumerate(state["sections"]):
        parts.append(f"## {i + 1}. {s['title']}\n\n{state['drafts'][i]['text']}\n")
    return {"report": "\n".join(parts)}


g = StateGraph(State)
g.add_node("plan", plan)
g.add_node("research", research)
g.add_node("check", check)
g.add_node("synthesize", synthesize)
g.add_edge(START, "plan")
g.add_conditional_edges("plan", dispatch, ["research"])
g.add_edge("research", "check")
g.add_conditional_edges("check", after_check, ["research", "synthesize"])
g.add_edge("synthesize", END)
app = g.compile()


def run(question, assign=True):
    return app.invoke({"question": question, "assign": assign})


def save(qid, st, mode):
    out = ROOT / "output"
    (out / "reports").mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = "" if mode == "기본" else {"단일": "_baseline", "배정 끔": "_noassign"}[mode]
    (out / "reports" / f"{qid}_{stamp}{suffix}.md").write_text(st["report"], encoding="utf-8")
    with open(out / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"qid": qid, "stamp": stamp, "mode": mode, **st}, ensure_ascii=False) + "\n")
    return stamp


if __name__ == "__main__":
    qs = {q["id"]: q["question"] for q in json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8"))}
    qid = sys.argv[1]
    st = run(qs.get(qid, qid))
    stamp = save(qid, st, "기본")

    for i, s in enumerate(st["sections"]):
        d = st["drafts"][i]
        print(f"절{i + 1} {s['title']} | {s['role']} | {d['round']}바퀴 {d['status']} | 읽음 {d['read']} | 리서처가 본 {d['researcher_chars']:,}자")
    print("코드가 고친 배정:", st["plan_fixes"] or "없음")
    print("두 번째 원고:", st["decisions"] or "없음")
    print(f"코디네이터가 본 글자 수: {st['coordinator_chars']:,}자")
    print(f"보고서 {len(st['report']):,}자 → output/reports/{qid}_{stamp}.md")
