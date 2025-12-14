package com.pdfanalyzer.core.integration.llm;

import com.pdfanalyzer.core.semantic.model.SemanticCell;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Component
public class LLMAdapter {

    private final RestTemplate restTemplate = new RestTemplate();
    private final com.pdfanalyzer.core.service.AnalysisCacheService cacheService;

    @Value("${gemini.api.key:}")
    private String apiKey;

    @Value("${gemini.model:gemini-pro}")
    private String modelName;

    private static final String GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/";

    public LLMAdapter(com.pdfanalyzer.core.service.AnalysisCacheService cacheService) {
        this.cacheService = cacheService;
    }

    /**
     * 한 번의 API 호출로 요약과 키워드를 함께 생성 (캐싱 포함)
     */
    public SummaryAndKeywords generateSummaryAndKeywords(List<SemanticCell> cells) {
        // 문서 내용으로 해시 생성
        String documentContent = cells.stream()
            .map(SemanticCell::getContent)
            .collect(Collectors.joining("\n"));
        String documentHash = cacheService.generateDocumentHash(documentContent);

        // 캐시 확인
        if (documentHash != null) {
            com.pdfanalyzer.core.service.AnalysisCacheService.CachedAnalysis cached = cacheService.get(documentHash);
            if (cached != null) {
                log.info("캐시된 분석 결과 사용 (API 호출 생략)");
                return new SummaryAndKeywords(cached.getSummary(), cached.getKeywords(), cached.getKeywordLocations());
            }
        }

        if (apiKey == null || apiKey.isEmpty()) {
            log.warn("⚠️ Gemini API 키가 설정되지 않았습니다. 폴백 요약을 생성합니다.");
            log.warn("환경변수 GEMINI_API_KEY를 설정하거나 application.yml에서 gemini.api.key를 설정하세요.");
            return new SummaryAndKeywords(
                generateFallbackSummary(cells),
                generateFallbackKeywords(cells)
            );
        }

        log.info("✅ Gemini API 키 확인 완료 (길이: {}자)", apiKey.length());

        try {
            // 구조 점수 통계 출력
            double avgScore = cells.stream()
                    .mapToDouble(SemanticCell::getStructuralScore)
                    .average()
                    .orElse(0.0);
            long highScoreCells = cells.stream()
                    .filter(c -> c.getStructuralScore() > 0.6)
                    .count();

            log.info("셀 통계 - 전체: {}, 평균 점수: {:.2f}, 높은 점수(>0.6): {}",
                    cells.size(), avgScore, highScoreCells);

            // 점수 기준을 낮춰서 더 많은 셀 포함
            List<SemanticCell> important = cells.stream()
                    .filter(c -> c.getStructuralScore() > 0.3)  // 0.6 -> 0.3으로 완화
                    .limit(30)  // 20 -> 30으로 증가
                    .collect(Collectors.toList());

            // 필터링된 셀이 없으면 모든 셀 사용
            if (important.isEmpty()) {
                log.warn("⚠️ 구조 점수가 0.3 이상인 셀이 없어 상위 30개 셀 사용");
                important = cells.stream().limit(30).collect(Collectors.toList());
            } else {
                log.info("✅ 선택된 셀: {}개 (점수 > 0.3)", important.size());
            }

            String prompt = buildCombinedPrompt(important);
            log.debug("프롬프트 내용: {}", prompt);
            String response = callGeminiAPI(prompt);

            if (response.isEmpty()) {
                return new SummaryAndKeywords(
                    generateFallbackSummary(cells),
                    generateFallbackKeywords(cells)
                );
            }

            return parseCombinedResponse(response, cells, documentHash);

        } catch (Exception e) {
            log.error("LLM 호출 실패", e);
            return new SummaryAndKeywords(
                generateFallbackSummary(cells),
                generateFallbackKeywords(cells)
            );
        }
    }

    @Deprecated
    public String generateSummary(List<SemanticCell> cells) {
        return generateSummaryAndKeywords(cells).summary;
    }

    @Deprecated
    public List<String> extractKeywords(List<SemanticCell> cells) {
        return generateSummaryAndKeywords(cells).keywords;
    }

    private String buildCombinedPrompt(List<SemanticCell> cells) {
        StringBuilder sb = new StringBuilder();
        sb.append("다음 문서를 분석하여 JSON 형식으로 응답해주세요.\n\n");
        sb.append("**중요**: 요약은 반드시 상세하고 포괄적이어야 하며, 내용 생략을 최소화해야 합니다.\n\n");
        sb.append("요구사항:\n");
        sb.append("1. 문서의 핵심 내용을 매우 상세히 요약 (summary):\n");
        sb.append("   - 문서의 전체 구조와 흐름을 파악하여 체계적으로 설명\n");
        sb.append("   - 문서의 주요 목적, 배경, 맥락을 구체적으로 기술\n");
        sb.append("   - 각 주요 섹션의 핵심 내용을 빠짐없이 요약 (5-7개 문단 이상)\n");
        sb.append("   - 구체적인 데이터, 수치, 통계, 날짜, 인명, 조직명 등을 반드시 포함\n");
        sb.append("   - 주요 논점, 주장, 근거, 결론을 명확하게 설명\n");
        sb.append("   - 세부 사항과 부연 설명도 포함하여 풍부하게 작성\n");
        sb.append("   - 결론, 시사점, 제언사항이 있다면 상세히 기술\n");
        sb.append("   - **최소 길이**: 500자 이상 (가능하면 800-1000자 권장)\n");
        sb.append("   - **원칙**: 문서의 주요 정보를 최대한 보존하고, 생략을 최소화할 것\n\n");
        sb.append("2. 핵심 키워드 10-20개 추출 (keywords 배열):\n");
        sb.append("   - 문서의 주제, 핵심 개념, 중요 용어를 대표하는 단어/구문\n");
        sb.append("   - 기술 용어, 전문 용어, 고유명사, 핵심 인물/조직명 포함\n");
        sb.append("   - 문서 전반에 걸쳐 중요하게 다뤄진 개념들을 모두 포함\n\n");
        sb.append("응답 형식 (JSON만 반환, 다른 텍스트 없이):\n");
        sb.append("{\n");
        sb.append("  \"summary\": \"문서의 매우 상세하고 포괄적인 요약 내용 (최소 500자 이상)...\",\n");
        sb.append("  \"keywords\": [\"키워드1\", \"키워드2\", \"키워드3\", ...]\n");
        sb.append("}\n\n");
        sb.append("문서 내용:\n\n");

        // 더 많은 셀을 포함하고, 내용을 충분히 길게 포함
        int cellCount = 0;
        for (SemanticCell cell : cells) {
            if (cellCount >= 50) break; // 최대 50개 셀로 증가

            // 헤더는 전체, 일반 셀은 800자까지 (이전 500자에서 증가)
            int maxLength = cell.isHeader() ? cell.getContent().length() : 800;
            String cellContent = cell.getContent().substring(0, Math.min(maxLength, cell.getContent().length()));

            if (cell.isHeader()) {
                sb.append("## ").append(cellContent).append("\n\n");
            } else {
                sb.append(cellContent).append("\n\n");
            }
            cellCount++;
        }
        return sb.toString();
    }

    private SummaryAndKeywords parseCombinedResponse(String response, List<SemanticCell> cells, String documentHash) {
        try {
            log.info("Gemini API 응답 내용: {}", response);

            // JSON 코드 블록 제거 (```json ... ```)
            response = response.replaceAll("```json\\s*", "").replaceAll("```\\s*$", "").trim();

            // JSON 응답 파싱 시도
            int summaryStart = response.indexOf("\"summary\"");
            int keywordsStart = response.indexOf("\"keywords\"");

            if (summaryStart != -1 && keywordsStart != -1) {
                // summary 추출 - "summary": " 다음부터 시작하여 ", 까지 (따옴표로 감싼 값 추출)
                int summaryValueStart = response.indexOf("\"", response.indexOf(":", summaryStart) + 1) + 1;
                int summaryValueEnd = summaryValueStart;

                // 따옴표로 감싼 문자열 끝을 찾기 (이스케이프된 따옴표는 무시)
                while (summaryValueEnd < response.length()) {
                    summaryValueEnd = response.indexOf("\"", summaryValueEnd);
                    if (summaryValueEnd == -1 || response.charAt(summaryValueEnd - 1) != '\\') {
                        break;
                    }
                    summaryValueEnd++;
                }

                String summary = response.substring(summaryValueStart, summaryValueEnd).trim();

                // keywords 추출
                int keywordsArrayStart = response.indexOf("[", keywordsStart);
                int keywordsArrayEnd = response.indexOf("]", keywordsArrayStart);
                String keywordsStr = response.substring(keywordsArrayStart + 1, keywordsArrayEnd);
                List<String> keywords = Arrays.stream(keywordsStr.split(","))
                    .map(k -> k.replaceAll("\"", "").trim())
                    .filter(k -> !k.isEmpty())
                    .limit(10)
                    .collect(Collectors.toList());

                // 키워드 위치 정보 추출
                Map<String, List<KeywordLocation>> keywordLocations = findKeywordLocations(keywords, cells);

                // 캐시에 저장
                if (documentHash != null) {
                    cacheService.put(documentHash, summary, keywords, keywordLocations);
                }

                log.info("파싱 성공 - summary 길이: {}, keywords 수: {}", summary.length(), keywords.size());
                return new SummaryAndKeywords(summary, keywords, keywordLocations);
            } else {
                log.warn("JSON 필드를 찾을 수 없음 - summaryStart: {}, keywordsStart: {}", summaryStart, keywordsStart);
            }
        } catch (Exception e) {
            log.warn("응답 파싱 실패, 폴백 사용", e);
        }

        log.info("폴백 요약 사용");
        return new SummaryAndKeywords(
            generateFallbackSummary(cells),
            generateFallbackKeywords(cells)
        );
    }

    public static class SummaryAndKeywords {
        public final String summary;
        public final List<String> keywords;
        public final Map<String, List<KeywordLocation>> keywordLocations;

        public SummaryAndKeywords(String summary, List<String> keywords) {
            this.summary = summary;
            this.keywords = keywords;
            this.keywordLocations = new HashMap<>();
        }

        public SummaryAndKeywords(String summary, List<String> keywords, Map<String, List<KeywordLocation>> keywordLocations) {
            this.summary = summary;
            this.keywords = keywords;
            this.keywordLocations = keywordLocations;
        }
    }

    public static class KeywordLocation {
        public final String cellId;
        public final String content;
        public final int pageNumber;
        public final double relevanceScore;

        public KeywordLocation(String cellId, String content, int pageNumber, double relevanceScore) {
            this.cellId = cellId;
            this.content = content;
            this.pageNumber = pageNumber;
            this.relevanceScore = relevanceScore;
        }
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private String callGeminiAPI(String prompt) {
        try {
            String url = GEMINI_API_URL + modelName + ":generateContent?key=" + apiKey;
            log.info("🌐 Gemini API 호출 시작 - 모델: {}, URL: {}", modelName, GEMINI_API_URL + modelName);
            log.debug("프롬프트 길이: {} 문자", prompt.length());

            Map<String, Object> requestBody = new HashMap<>();
            Map<String, Object> content = new HashMap<>();
            Map<String, String> part = new HashMap<>();
            part.put("text", prompt);
            content.put("parts", Collections.singletonList(part));
            requestBody.put("contents", Collections.singletonList(content));

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestBody, headers);

            log.info("📤 API 요청 전송 중...");
            long startTime = System.currentTimeMillis();

            ResponseEntity<Map> response = restTemplate.exchange(
                    url,
                    HttpMethod.POST,
                    entity,
                    Map.class
            );

            long duration = System.currentTimeMillis() - startTime;
            log.info("📥 API 응답 수신 완료 - 상태: {}, 소요시간: {}ms", response.getStatusCode(), duration);

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                String extracted = extractTextFromResponse(response.getBody());
                if (extracted != null && !extracted.isEmpty()) {
                    log.info("✅ 응답 텍스트 추출 성공 - 길이: {} 문자", extracted.length());
                    return extracted;
                } else {
                    log.error("❌ 응답 본문이 비어있습니다. 응답 내용: {}", response.getBody());
                    return "";
                }
            }

            log.warn("⚠️ Gemini API 응답이 비정상입니다: {}, 본문: {}", response.getStatusCode(), response.getBody());
            return "";

        } catch (org.springframework.web.client.HttpClientErrorException e) {
            log.error("❌ Gemini API HTTP 클라이언트 오류 - 상태: {}, 메시지: {}, 본문: {}",
                    e.getStatusCode(), e.getMessage(), e.getResponseBodyAsString());
            return "";
        } catch (org.springframework.web.client.HttpServerErrorException e) {
            log.error("❌ Gemini API 서버 오류 - 상태: {}, 메시지: {}, 본문: {}",
                    e.getStatusCode(), e.getMessage(), e.getResponseBodyAsString());
            return "";
        } catch (Exception e) {
            log.error("❌ Gemini API 호출 중 예상치 못한 오류 발생", e);
            log.error("오류 타입: {}, 메시지: {}", e.getClass().getName(), e.getMessage());
            return "";
        }
    }

    @SuppressWarnings("unchecked")
    private String extractTextFromResponse(Map<String, Object> responseBody) {
        try {
            List<Map<String, Object>> candidates = (List<Map<String, Object>>) responseBody.get("candidates");
            if (candidates != null && !candidates.isEmpty()) {
                Map<String, Object> content = (Map<String, Object>) candidates.get(0).get("content");
                if (content != null) {
                    List<Map<String, Object>> parts = (List<Map<String, Object>>) content.get("parts");
                    if (parts != null && !parts.isEmpty()) {
                        return (String) parts.get(0).get("text");
                    }
                }
            }
        } catch (Exception e) {
            log.error("응답 파싱 중 오류", e);
        }
        return "";
    }

    private String generateFallbackSummary(List<SemanticCell> cells) {
        StringBuilder sb = new StringBuilder();

        sb.append("=== 문서 요약 ===\n\n");

        // 헤더와 내용 셀을 분리
        List<SemanticCell> headers = cells.stream()
                .filter(SemanticCell::isHeader)
                .sorted((a, b) -> Double.compare(b.getStructuralScore(), a.getStructuralScore()))
                .limit(5)
                .collect(Collectors.toList());

        List<SemanticCell> contentCells = cells.stream()
                .filter(c -> !c.isHeader())
                .filter(c -> c.getContent().length() > 50) // 최소 50자 이상
                .sorted((a, b) -> Double.compare(b.getStructuralScore(), a.getStructuralScore()))
                .limit(20)
                .collect(Collectors.toList());

        // 주요 섹션과 내용을 최대한 상세하게 포함
        int addedContent = 0;

        // 모든 헤더와 관련 내용을 포함 (최대 10개 섹션)
        for (int i = 0; i < Math.min(headers.size(), 10); i++) {
            // 헤더 출력
            if (i < headers.size()) {
                sb.append("\n【 ").append(headers.get(i).getContent()).append(" 】\n");
            }

            // 해당 헤더 관련 내용 찾기 (섹션당 3-5개 내용 셀)
            int contentStart = i * 3;
            int contentEnd = Math.min(contentStart + 5, contentCells.size());

            for (int j = contentStart; j < contentEnd && addedContent < 40; j++) {  // 최대 40개로 증가
                SemanticCell cell = contentCells.get(j);
                String content = cell.getContent().trim();

                // 의미있는 부분을 더 길게 추출 (400자로 증가)
                String summary = extractMeaningfulContent(content, 400);
                if (!summary.isEmpty()) {
                    sb.append("• ").append(summary).append("\n");
                    addedContent++;
                }
            }
        }

        // 내용이 부족하면 추가 내용 보강
        if (addedContent < 10) {
            sb.append("\n【 추가 주요 내용 】\n");

            // 아직 사용하지 않은 content cell들을 추가
            int startIdx = addedContent * 3;
            for (int i = startIdx; i < Math.min(startIdx + 20, contentCells.size()); i++) {
                String excerpt = extractMeaningfulContent(contentCells.get(i).getContent(), 300);
                if (!excerpt.isEmpty()) {
                    sb.append("• ").append(excerpt).append("\n");
                    addedContent++;
                }
            }
        }

        // 문서 구조도 함께 보여주기
        if (headers.size() > 5) {
            sb.append("\n【 문서 전체 구조 】\n");
            List<SemanticCell> allHeaders = cells.stream()
                    .filter(SemanticCell::isHeader)
                    .limit(15)  // 최대 15개 헤더
                    .collect(Collectors.toList());

            for (int i = 0; i < allHeaders.size(); i++) {
                sb.append((i + 1)).append(". ").append(allHeaders.get(i).getContent()).append("\n");
            }
        }

        // 통계 정보
        long headerCount = cells.stream().filter(SemanticCell::isHeader).count();
        long contentCellCount = cells.size() - headerCount;
        sb.append("\n【 문서 정보 】\n");
        sb.append("• 총 ").append(cells.size()).append("개 섹션 (제목: ").append(headerCount)
          .append(", 내용: ").append(contentCellCount).append(")\n");

        int estimatedPages = cells.stream()
                .mapToInt(c -> c.getPageNumber() != null ? c.getPageNumber() : 0)
                .max()
                .orElse(1);
        sb.append("• 예상 페이지 수: 약 ").append(estimatedPages).append(" 페이지\n");

        return sb.toString();
    }

    /**
     * 내용에서 의미있는 부분 추출
     */
    private String extractMeaningfulContent(String content, int maxLength) {
        if (content == null || content.trim().isEmpty()) return "";

        content = content.trim();

        // 너무 짧으면 그대로 반환
        if (content.length() <= maxLength) {
            return content;
        }

        // maxLength까지 자르되, 문장이 끊기지 않도록 마지막 마침표/줄바꿈 찾기
        String truncated = content.substring(0, maxLength);

        // 마지막 문장 구분자 찾기
        int lastPeriod = Math.max(
            truncated.lastIndexOf(". "),
            Math.max(truncated.lastIndexOf(".\n"), truncated.lastIndexOf("。"))
        );

        if (lastPeriod > maxLength / 2) {
            // 문장이 너무 짧지 않으면 문장 단위로 자르기
            return truncated.substring(0, lastPeriod + 1).trim();
        }

        // 그냥 자르고 "..." 추가
        return truncated.trim() + "...";
    }

    private List<String> generateFallbackKeywords(List<SemanticCell> cells) {
        List<String> keywords = new ArrayList<>();

        // 헤더에서 키워드 추출
        cells.stream()
                .filter(SemanticCell::isHeader)
                .map(SemanticCell::getContent)
                .map(String::trim)
                .filter(s -> s.length() > 2 && s.length() < 100)
                .limit(5)
                .forEach(keywords::add);

        // 중요도 높은 셀에서 긴 단어 추출
        cells.stream()
                .filter(c -> !c.isHeader() && c.getStructuralScore() > 0.6)
                .flatMap(c -> Arrays.stream(c.getContent().split("\\s+")))
                .filter(word -> word.length() > 4)  // 4자 이상
                .map(word -> word.replaceAll("[^가-힣a-zA-Z0-9]", ""))  // 특수문자 제거
                .filter(word -> word.length() > 4)
                .distinct()
                .limit(10)
                .forEach(keywords::add);

        // 최소 5개 보장
        if (keywords.size() < 5) {
            keywords.add("문서분석");
            keywords.add("PDF");
            keywords.add("자동추출");
        }

        return keywords.stream().distinct().limit(15).collect(Collectors.toList());
    }

    /**
     * 키워드가 등장하는 셀 위치를 찾습니다
     */
    private Map<String, List<KeywordLocation>> findKeywordLocations(List<String> keywords, List<SemanticCell> cells) {
        Map<String, List<KeywordLocation>> locations = new HashMap<>();

        for (String keyword : keywords) {
            List<KeywordLocation> keywordLocs = new ArrayList<>();

            for (int i = 0; i < cells.size(); i++) {
                SemanticCell cell = cells.get(i);
                String content = cell.getContent().toLowerCase();
                String keywordLower = keyword.toLowerCase();

                // 키워드가 셀 내용에 포함되어 있는지 확인
                if (content.contains(keywordLower)) {
                    // 관련도 점수 계산 (키워드 출현 빈도 + 구조 점수)
                    int occurrences = countOccurrences(content, keywordLower);
                    double relevanceScore = (occurrences * 0.5) + (cell.getStructuralScore() * 0.5);

                    KeywordLocation location = new KeywordLocation(
                        cell.getId(),
                        cell.getContent().substring(0, Math.min(300, cell.getContent().length())), // 처음 300자
                        i + 1, // 페이지 번호 근사치 (실제로는 셀 순서)
                        relevanceScore
                    );
                    keywordLocs.add(location);
                }
            }

            // 관련도 점수로 정렬 (높은 순)
            keywordLocs.sort((a, b) -> Double.compare(b.relevanceScore, a.relevanceScore));

            // 상위 5개만 유지
            if (keywordLocs.size() > 5) {
                keywordLocs = keywordLocs.subList(0, 5);
            }

            locations.put(keyword, keywordLocs);
        }

        return locations;
    }

    private int countOccurrences(String text, String keyword) {
        int count = 0;
        int index = 0;
        while ((index = text.indexOf(keyword, index)) != -1) {
            count++;
            index += keyword.length();
        }
        return count;
    }
}
