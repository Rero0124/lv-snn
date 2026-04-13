"""학습 스크립트 유틸리티"""

def decompose_jamo(text):
    """한글 음절을 자모로 분해"""
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
    return set(result)

def jamo_overlap(output, expected):
    """출력과 정답의 자모 겹침 비율 (0.0~1.0)"""
    out_jamo = decompose_jamo(output)
    exp_jamo = decompose_jamo(expected)
    if not exp_jamo:
        return 0.0
    return len(out_jamo & exp_jamo) / len(exp_jamo)
