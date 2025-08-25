# Semantic Deduplicator - 의미적 중복 제거 시스템

## 사용 방법

### 1. 필요 패키지 설치
```bash
pip install sentence-transformers scikit-learn numpy torch Levenshtein
```

### 2. 테스트 실행 방법
`duplica.py` 파일 하단의 `sample_text` 부분을 수정하여 테스트:

```python
# 파일을 열어서 하단 부분 수정
if __name__ == "__main__":
    
    sample_text = """
    텍스트삽입.
    """
    
    # 아래 코드는 그대로 두고 실행
    deduplicator = SemanticDeduplicator()
    cleaned_text, metrics = deduplicator.deduplicate(sample_text)
    ...
```

## 사용된 기술

### 3단계 중복 제거 시스템
1. **Level 1 - 표면적 유사도 (95% 임계값)**
   - MD5 해싱으로 완전 중복 제거
   - Levenshtein 편집 거리로 유사 문장 탐지

2. **Level 2 - 의미적 유사도 (90% 임계값)**  
   - 한국어 SBERT 모델 (`jhgan/ko-sroberta-multitask`)
   - 문장 임베딩 후 코사인 유사도 계산

3. **Level 3 - 문맥적 유사도 (85% 임계값)**
   - 주변 문장 컨텍스트 고려
   - 위치 인코딩으로 문서 내 위치 반영

### 핵심 라이브러리
- `sentence-transformers`: 문장을 벡터로 변환
- `scikit-learn`: 유사도 계산
- `numpy`: 벡터 연산
- `torch`: 딥러닝 백엔드
- `Levenshtein`: 문자열 유사도
