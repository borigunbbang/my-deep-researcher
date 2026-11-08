import json

import streamlit as st
import graph


def show(result):
    tab_report, tab_sections = st.tabs(["보고서", "절마다 누가 무엇을"])
    with tab_report:
        st.markdown(result["report"])
    with tab_sections:
        st.caption(f"코디네이터가 본 글자 수 {result['coordinator_chars']:,}자 — 원문은 리서처 아래에 남고, 위로는 원고만 올라온다")
        for fix in result["plan_fixes"]:
            st.warning(f"코드가 고친 배정: {fix}")
        for note in result["decisions"]:
            st.info(f"점검: {note}")
        for i, sec in enumerate(result["sections"]):
            d = result["drafts"].get(i) or result["drafts"].get(str(i))  # 방금 실행은 숫자 키, 파일에서 읽으면 문자열 키
            mine = [r for r in result["results"] if r["idx"] == i]
            seen = d.get("researcher_chars") or sum(r["researcher_chars"] for r in mine)
            with st.expander(f"{i + 1}. {sec['title']} — {sec['role']}"):
                st.write(f"시작 문서 `{sec['start_doc']}` · 예산 {sec.get('budget', '-')}건 · {d['round']}바퀴 · 리서처가 본 글자 수 {seen:,}자")
                st.write("읽은 순서: " + " → ".join(d["read"]))
                st.write(f"신고: **{d['status']}** — {d['reason']}")
                st.markdown("**올린 원고**")
                st.markdown(d["text"])
                dropped = [x for r in mine for x in r.get("dropped", [])]  # 초기 대조군 기록엔 없다
                fixed = [x for r in mine for x in r.get("fixed", [])]
                if dropped or fixed:
                    st.markdown(f"**코드가 걸러 낸 줄 {len(dropped)}개 · 제목을 바로잡은 줄 {len(fixed)}개**")
                    for x in dropped:
                        st.caption(f"✂️ {x['why']}: {x['line']}")
                    for x in fixed:
                        st.caption(f"🔁 «{x['from']}» → «{x['to']}»: {x['line']}")


st.title("오사카·교토 딥리서처")
page = st.sidebar.radio("보기", ["새 질문", "지난 실행"])

if page == "새 질문":
    q = st.text_input("질문")
    if st.button("실행") and q:
        with st.spinner("코디네이터가 목차를 짜고 리서처들이 읽는 중… (1~3분)"):
            st.session_state.result = graph.run(q)
        graph.save("demo", st.session_state.result, "기본")
    if "result" in st.session_state:
        show(st.session_state.result)
else:
    lines = (graph.ROOT / "output/runs.jsonl").read_text(encoding="utf-8").splitlines()
    runs = [r for r in map(json.loads, lines) if "mode" in r][::-1]  # mode 없는 초기 개발 기록은 뺀다, 최신이 위
    labels = [f"{r['qid']} · {r['mode']} · {r['stamp']}" for r in runs]
    pick = st.selectbox("실행 기록 (기본 · 배정 끔 · 단일)", range(len(runs)), format_func=lambda i: labels[i])
    st.write(f"**질문:** {runs[pick]['question']}")
    show(runs[pick])
