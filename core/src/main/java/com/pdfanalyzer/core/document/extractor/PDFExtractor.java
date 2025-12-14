package com.pdfanalyzer.core.document.extractor;

import com.pdfanalyzer.core.document.model.DocumentMetadata;
import lombok.extern.slf4j.Slf4j;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * PDF 문서에서 텍스트와 기본 구조 정보를 추출하는 컴포넌트
 */
@Slf4j
@Component
public class PDFExtractor {

    /**
     * PDF 파일에서 텍스트를 페이지별로 추출
     * 향상된 레이아웃 및 포맷 보존
     */
    public List<String> extractTextByPages(File pdfFile) throws IOException {
        log.info("PDF 텍스트 추출 시작: {}", pdfFile.getName());

        List<String> pages = new ArrayList<>();

        try (PDDocument document = Loader.loadPDF(pdfFile)) {
            PDFTextStripper stripper = new PDFTextStripper();

            // 레이아웃 및 포맷 보존 설정
            stripper.setSortByPosition(true);  // 위치 기반 정렬로 레이아웃 보존
            stripper.setShouldSeparateByBeads(true);  // 논리적 섹션 구분
            stripper.setLineSeparator("\n");  // 명확한 줄바꿈
            stripper.setWordSeparator(" ");  // 단어 구분자
            stripper.setParagraphStart("");  // 문단 시작
            stripper.setParagraphEnd("\n\n");  // 문단 끝 (두 줄 개행)

            int totalPages = document.getNumberOfPages();

            for (int i = 1; i <= totalPages; i++) {
                stripper.setStartPage(i);
                stripper.setEndPage(i);
                String pageText = stripper.getText(document);

                // 불필요한 공백 정리 (하지만 구조는 유지)
                pageText = cleanupText(pageText);

                pages.add(pageText);
                log.debug("페이지 {} 추출 완료: {} 문자", i, pageText.length());
            }

            log.info("총 {} 페이지 추출 완료", totalPages);
        }

        return pages;
    }

    /**
     * 텍스트 정리 (구조는 유지하면서 불필요한 공백 제거)
     */
    private String cleanupText(String text) {
        if (text == null) return "";

        // 3개 이상의 연속된 공백을 2개로 축소
        text = text.replaceAll(" {3,}", "  ");

        // 4개 이상의 연속된 줄바꿈을 3개로 축소
        text = text.replaceAll("\n{4,}", "\n\n\n");

        // 줄 끝의 공백 제거
        text = text.replaceAll(" +\n", "\n");

        return text.trim();
    }

    /**
     * PDF 파일에서 전체 텍스트를 한 번에 추출
     */
    public String extractFullText(File pdfFile) throws IOException {
        log.info("PDF 전체 텍스트 추출 시작: {}", pdfFile.getName());

        try (PDDocument document = Loader.loadPDF(pdfFile)) {
            PDFTextStripper stripper = new PDFTextStripper();

            // 레이아웃 및 포맷 보존 설정
            stripper.setSortByPosition(true);
            stripper.setShouldSeparateByBeads(true);
            stripper.setLineSeparator("\n");
            stripper.setWordSeparator(" ");
            stripper.setParagraphStart("");
            stripper.setParagraphEnd("\n\n");

            String text = stripper.getText(document);
            text = cleanupText(text);

            log.info("텍스트 추출 완료: {} 문자", text.length());
            return text;
        }
    }

    /**
     * InputStream에서 PDF 텍스트 추출
     */
    public String extractFullText(InputStream inputStream) throws IOException {
        log.info("InputStream에서 PDF 텍스트 추출 시작");

        byte[] pdfBytes = inputStream.readAllBytes();
        try (PDDocument document = Loader.loadPDF(pdfBytes)) {
            PDFTextStripper stripper = new PDFTextStripper();

            // 레이아웃 및 포맷 보존 설정
            stripper.setSortByPosition(true);
            stripper.setShouldSeparateByBeads(true);
            stripper.setLineSeparator("\n");
            stripper.setWordSeparator(" ");
            stripper.setParagraphStart("");
            stripper.setParagraphEnd("\n\n");

            String text = stripper.getText(document);
            text = cleanupText(text);

            log.info("텍스트 추출 완료: {} 문자", text.length());
            return text;
        }
    }

    /**
     * PDF 메타데이터 추출
     */
    public DocumentMetadata extractMetadata(File pdfFile) throws IOException {
        log.info("PDF 메타데이터 추출: {}", pdfFile.getName());

        try (PDDocument document = Loader.loadPDF(pdfFile)) {
            DocumentMetadata metadata = new DocumentMetadata();
            metadata.setPageCount(document.getNumberOfPages());

            if (document.getDocumentInformation() != null) {
                metadata.setTitle(document.getDocumentInformation().getTitle());
                metadata.setAuthor(document.getDocumentInformation().getAuthor());
                metadata.setSubject(document.getDocumentInformation().getSubject());
            }

            log.info("메타데이터 추출 완료: {} 페이지", metadata.getPageCount());
            return metadata;
        }
    }
}
