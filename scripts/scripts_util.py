"""학습 스크립트 유틸리티"""

from collections import Counter


def decompose_jamo_list(text):
    """한글 음절을 자모 리스트로 분해 (중복 보존)"""
    result = []
    for c in text:
        code = ord(c)
        if 0xAC00 <= code <= 0xD7A3:
            # 완성형 한글
            code -= 0xAC00
            cho = code // (21 * 28)
            jung = (code % (21 * 28)) // 28
            jong = code % 28
            CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
            JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
            JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
            result.append(CHO[cho])
            result.append(JUNG[jung])
            if JONG[jong]:
                result.append(JONG[jong])
        elif 0x3131 <= code <= 0x3163:
            # 이미 자모
            result.append(c)
    return result

def decompose_jamo(text):
    """한글 음절을 자모 집합으로 분해 (중복 제거)"""
    return set(decompose_jamo_list(text))

def jamo_overlap(output, expected):
    """출력과 정답의 자모 겹침 비율 (0.0~1.0), 중복(multiset) 고려.

    각 자모는 정답에 등장한 횟수까지만 가산점 — 같은 자모를 그 이상 반복해도
    초과분은 0점이다. (예: '안녕'의 ㅇ·ㄴ 은 각각 2번까지만 인정,
    'ㅇㅇㅇㅇㅇㅇ' 같은 도배는 ㅇ 2개분만 인정.)
    """
    out_c = Counter(decompose_jamo_list(output))
    exp_c = Counter(decompose_jamo_list(expected))
    total = sum(exp_c.values())
    if total == 0:
        return 0.0
    matched = sum((out_c & exp_c).values())  # 자모별 min(출력, 정답) 합 = 초과분 자동 제외
    return matched / total


def is_complete_syllable(ch):
    """완성형 한글 음절(조합된 한 글자)인가"""
    return len(ch) == 1 and 0xAC00 <= ord(ch) <= 0xD7A3


def completion_score(output, expected, partial=0.5):
    """완성형(조합된) 음절에 주는 보너스 점수 (0.0~1.0).

    낱자모(ㅇㅏㄴ)가 아니라 실제 음절(안)을 만들었을 때 가산한다.
    출력의 완성형 음절마다 정답 음절들과 비교해 best 점수를 취함:
      • 음절이 정확히 같으면 1.0
      • 자모는 같은데 순서만 다르면(예 안↔낭) partial × 자모겹침(순서 무시)
    음절별 best 점수의 합을 정답 음절 수로 정규화(최대 1.0).
    완성형이 아닌 낱자모는 이 보너스에 기여하지 않는다.
    """
    out_syls = [c for c in output if is_complete_syllable(c)]
    exp_syls = [c for c in expected if is_complete_syllable(c)]
    if not out_syls or not exp_syls:
        return 0.0
    total = 0.0
    for o in out_syls:
        best = 0.0
        for e in exp_syls:
            s = 1.0 if o == e else partial * jamo_overlap(o, e)
            if s > best:
                best = s
        total += best
    return min(1.0, total / len(exp_syls))
