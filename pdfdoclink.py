if __name__ == "__main__":
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.models.tesseract_ocr_model import TesseractOcrOptions

    # Thiết lập pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = False
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # Chế độ chính xác

    pipeline_options.ocr_options = TesseractOcrOptions()

    # Khởi tạo DocumentConverter với các cấu hình PDF
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Đường dẫn tệp PDF
    pdf_path = "Data/04_2007_QH12_59652.pdf"
    try:
        # Chuyển đổi PDF
        result = converter.convert(pdf_path)

        # Xuất kết quả ra tệp Markdown
        output_text = result.document.export_to_markdown()
        with open("04_2007_QH12_59652.md", "w") as f:
            f.write(output_text)

        print("Conversion completed successfully!")
    except Exception as e:
        print(f"Error occurred: {e}")
