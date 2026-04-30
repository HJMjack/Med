#!/usr/bin/env python3
"""
뇌졸중 응급 평가 갭 분석기
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
임상 시나리오를 입력하면, ischemic_stroke_er_algorithm.json
(v3.0)의 4대 카테고리 체크리스트를 기준으로
"확인된 정보" vs "아직 빠진 정보"를 분석합니다.

사용법:
  export ANTHROPIC_API_KEY="sk-ant-..."

  # 대화형 모드
  python3 stroke_gap_analyzer.py

  # 시나리오를 직접 인자로
  python3 stroke_gap_analyzer.py "62세 여자, 3시간 전 갑자기 말이 안 나오고..."

요구사항: Python 3.8+, requests (표준 설치)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import os
import sys
import requests

# ── 설정 ────────────────────────────────────────────────────
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALGO_PATH = os.path.join(SCRIPT_DIR, "ischemic_stroke_er_algorithm.json")


# ── 알고리듬 로드 및 참조 데이터 추출 ──────────────────────
def load_algorithm(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_reference(algo: dict) -> dict:
    """알고리듬 JSON에서 4대 카테고리 핵심 항목만 추출."""
    def get_step(sid):
        return next((s for s in algo["steps"] if s["id"] == sid), {})

    S3 = get_step("S3")
    S4 = get_step("S4")
    S5 = get_step("S5")
    S6 = get_step("S6")

    return {
        "history_taking": {
            "checklist": [
                f"[{i['priority']}] {i['item']}"
                for i in S3.get("history_checklist", [])
            ],
            "etiology_clues": [
                f"{c['clue']} -> {c['suggests']}"
                for c in S3.get("etiology_clue_history", {}).get("clues", [])
            ],
            "stroke_mimic_questions": S3.get(
                "stroke_mimics_rapid_screen", {}
            ).get("questions", []),
            "medication_tpa_relevance": S3.get("medication_tpa_relevance", {}),
        },
        "neurologic_exam": {
            "initial_overview": S4.get("initial_overview", {}).get("signs", []),
            "vital_signs_required": S4.get("vital_signs_in_exam", {}).get("required", []),
            "nihss_items": [
                f"{i['number']} {i['name']} ({i['range']}점)"
                for i in S4.get("nihss", {}).get("items", [])
            ],
            "nihss_thresholds": S4.get("nihss", {}).get("treatment_thresholds", {}),
            "nihss_limitations": S4.get("nihss", {}).get("nihss_limitations", []),
            "posterior_HINTS": S4.get("posterior_circulation_exam", {}).get("HINTS_exam", {}),
            "non_dominant_items": S4.get("non_dominant_hemisphere_exam", {}).get("items", []),
            "dysphagia_screening": S4.get("dysphagia_screening", {}),
        },
        "imaging": {
            "acute_protocol": [
                {
                    "modality": m["modality"],
                    "purpose": m["purpose"],
                    "key_findings": m.get("findings_to_assess",
                                         m.get("target_vessels",
                                         m.get("indications", []))),
                }
                for m in S5.get("imaging_protocol", [])
            ],
            "inpatient_followup": [
                f"{m['name']}: {m['purpose']}"
                for m in S5.get("inpatient_follow_up_imaging", {}).get("modalities", [])
            ],
        },
        "laboratory": {
            "essential": [
                f"{t['test']}: {t['threshold']}"
                for t in S6.get("essential_tests", [])
            ],
            "additional": [
                f"{t['test']} ({t['indication']})"
                for t in S6.get("additional_tests", [])
            ],
            "inpatient_etiology": [
                t["test"]
                for t in S6.get("inpatient_workup_for_etiology", {}).get("tests", [])
            ],
            "toast_linked_workup": S6.get("inpatient_workup_for_etiology", {}).get(
                "toast_linked_workup", {}
            ),
        },
    }


# ── 시스템 프롬프트 ─────────────────────────────────────────
def build_system_prompt(reference: dict) -> str:
    ref_json = json.dumps(reference, ensure_ascii=False, indent=2)
    return f"""당신은 허혈성 뇌졸중 응급 평가 갭(gap) 분석 보조 시스템입니다.

아래 알고리듬 참조 데이터를 기준으로, 사용자가 제시하는 임상 시나리오를 분석하여
"이미 확인된 정보"와 "아직 확인되지 않은 정보(갭)"를 정확히 파악합니다.

══════════════════════════════════════════════
알고리듬 참조 데이터 (ischemic_stroke_er_algorithm v3.0)
══════════════════════════════════════════════
{ref_json}

══════════════════════════════════════════════
분석 원칙
══════════════════════════════════════════════
1. 시나리오에 명시적으로 기술된 항목 -> 확인됨 (V 표시)
2. 시나리오에서 암시되거나 추론 가능하지만 불확실한 항목 -> 갭으로 처리
3. 언급 없거나 불확실한 항목 -> 우선순위별 갭으로 분류:
   - [즉시] tPA/EVT 투여 결정에 직결되는 항목
   - [중요] 치료 계획·합병증 예방에 필요한 항목
   - [입원후] 이차 예방·원인 분류(TOAST) 목적의 항목
4. 각 갭 항목에는 왜 그 정보가 필요한지 임상 이유를 한 줄로 덧붙일 것
5. 이 시나리오에서 특히 주목해야 할 임상 맥락을 강조할 것
   (예: "가슴 두근거림 -> 심방세동 가능성 -> 항응고제 복용 여부·INR 확인 필수")

══════════════════════════════════════════════
출력 형식 (반드시 준수)
══════════════════════════════════════════════

## [V] 확인된 정보
(카테고리별로 간결하게 나열)

## [즉시] 지금 당장 확인 필요 — tPA·EVT 결정
(항목마다: 확인 방법 + 임상 이유)

## [중요] 추가 확인 필요 — 치료 계획
(항목마다: 확인 방법 + 임상 이유)

## [입원후] 입원 후 확인 — 이차 예방·원인 분류
(항목마다: 확인 방법 + 임상 이유)

## [포인트] 이 시나리오의 핵심 임상 고려사항
(이 증례에서 특히 중요한 2-3가지를 서술)

모든 출력은 한국어로 작성하세요."""


# ── Anthropic API 호출 (streaming) ─────────────────────────
def call_claude_stream(system_prompt: str, user_message: str, api_key: str):
    """Anthropic Messages API를 streaming으로 호출."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "accept": "text/event-stream",
    }
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system_prompt,
        "stream": True,
        "messages": [
            {"role": "user", "content": f"다음 임상 시나리오를 분석해주세요:\n\n{user_message}"}
        ],
    }

    with requests.post(
        ANTHROPIC_API_URL, headers=headers, json=payload, stream=True, timeout=90
    ) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"API 오류 {resp.status_code}: {resp.text[:400]}")

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    yield delta.get("text", "")


# ── 시나리오 분석 출력 ───────────────────────────────────────
def analyze(scenario: str, system_prompt: str, api_key: str):
    print()
    print("=" * 62)
    print("  갭 분석 결과")
    print("=" * 62)
    print()
    try:
        for chunk in call_claude_stream(system_prompt, scenario, api_key):
            print(chunk, end="", flush=True)
    except RuntimeError as e:
        print(f"\n[오류] {e}")
    print("\n")


# ── 멀티라인 입력 헬퍼 ──────────────────────────────────────
def read_scenario_interactive() -> str:
    """Enter 두 번 연속 입력 시 입력 완료."""
    print()
    print("-" * 62)
    print("시나리오를 입력하세요.")
    print("(완료: Enter 두 번  |  종료: Ctrl+C)")
    print("-" * 62)
    lines = []
    empty_streak = 0
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            empty_streak += 1
            if empty_streak >= 2:
                break
        else:
            empty_streak = 0
            lines.append(line)
    return "\n".join(lines).strip()


# ── 진입점 ──────────────────────────────────────────────────
def main():
    # API 키 확인
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print()
        print("오류: ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
        print()
        print("  설정 방법 (터미널):")
        print("    export ANTHROPIC_API_KEY='sk-ant-...'")
        print()
        print("  또는 실행 시 인라인으로:")
        print("    ANTHROPIC_API_KEY='sk-ant-...' python3 stroke_gap_analyzer.py")
        sys.exit(1)

    # 알고리듬 파일 확인
    if not os.path.exists(ALGO_PATH):
        print(f"오류: 알고리듬 파일 없음\n  경로: {ALGO_PATH}")
        sys.exit(1)

    algo = load_algorithm(ALGO_PATH)
    reference = build_reference(algo)
    system_prompt = build_system_prompt(reference)
    algo_version = algo.get("meta", {}).get("version", "?")

    print()
    print("+----------------------------------------------------------+")
    print("|  뇌졸중 응급 평가 갭 분석기                              |")
    print(f"|  알고리듬 v{algo_version} | 모델: {MODEL:<28}|")
    print("|  종료: Ctrl+C                                            |")
    print("+----------------------------------------------------------+")

    # 인자로 시나리오가 주어진 경우 (단발 실행)
    if len(sys.argv) > 1:
        scenario = " ".join(sys.argv[1:])
        print(f"\n[시나리오]\n{scenario}")
        analyze(scenario, system_prompt, api_key)
        return

    # 대화형 루프
    while True:
        try:
            scenario = read_scenario_interactive()
            if not scenario:
                print("시나리오가 비어 있습니다. 다시 입력하세요.")
                continue
            print(f"\n[입력된 시나리오]\n{scenario}")
            analyze(scenario, system_prompt, api_key)
        except KeyboardInterrupt:
            print("\n\n종료합니다.")
            break


if __name__ == "__main__":
    main()
