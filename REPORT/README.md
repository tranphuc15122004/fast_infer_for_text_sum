# ACL LaTeX Report Project

Thư mục này chứa template và mã nguồn LaTeX báo cáo khoa học theo định dạng chuẩn của hội nghị **ACL (Association for Computational Linguistics)**.

## Cấu trúc thư mục

```text
REPORT/
├── main.tex           # File LaTeX chính (entrypoint để biên dịch)
├── acl.sty            # File định kiểu (style file) chuẩn ACL
├── acl_natbib.bst     # File định dạng trích dẫn BibTeX chuẩn ACL
├── custom.bib         # File chứa danh mục tài liệu trích dẫn (BibTeX)
├── figures/           # Thư mục chứa hình ảnh, biểu đồ
├── Makefile           # Makefile điều khiển biên dịch qua make
├── build.sh           # Script bash biên dịch tiện lợi
├── .latexmkrc         # Cấu hình tự động cho latexmk (pdflatex + bibtex)
├── .gitignore         # Bỏ qua các file trung gian khi compile
└── README.md          # Hướng dẫn này
```

## Cách biên dịch ra PDF

Hệ thống đã có sẵn `latexmk`, `pdflatex`, `bibtex` (TeX Live / TinyTeX). Bạn có thể biên dịch bằng bất kỳ cách nào dưới đây:

### Cách 1: Sử dụng Makefile (Khuyên dùng)

```bash
cd REPORT

# Biên dịch ra main.pdf
make

# Dọn dẹp các file rác trung gian (.aux, .log, .bbl, ...)
make clean

# Dọn dẹp toàn bộ kể cả main.pdf
make distclean

# Chế độ theo dõi file tự động biên dịch lại khi sửa (watch mode)
make watch
```

### Cách 2: Sử dụng script `build.sh`

```bash
# Biên dịch
bash REPORT/build.sh

# Dọn dẹp file trung gian
bash REPORT/build.sh clean
```

### Cách 3: Sử dụng trực tiếp `latexmk`

```bash
cd REPORT
latexmk main.tex
```

## Ghi chú định dạng ACL
- Mặc định văn bản dùng chế độ review (có đánh số dòng, ẩn tác giả): `\usepackage[review]{acl}`.
- Để nộp bản preprint (có số trang, hiện tên tác giả): sửa thành `\usepackage[preprint]{acl}`.
- Để nộp bản camera-ready/final: sửa thành `\usepackage[final]{acl}`.

## Định hướng của báo cáo
- Báo cáo này sẽ thực hiện viết trình bày như vừa là một report và paper nghiên cứu khoa học
- Đề tài thực hiện là "Tăng tốc inference cho tác vụ long context, tóm tắt văn bản". 
- Mục tiêu là báo cáo về kết quả nghiên cứu , thực hiện được trong 3 tháng qua thực hiện đề tài . Có một số các công việc được thử nghiệm ở /home/tuantb/my_GrowMTP/ và /home/tuantb/fast_infer_text_sum_Viet/ , không nên báo cáo vào phần nội dung chính vì nó chưa có kết quả thực hiện.
- Method chính của tôi sẽ chọn để báo cáo để bảo cáo là phương pháp Training Free đang thực hiện thực nghiệm
- Hiện tại đang thực hiện các analysis để điều chỉnh phương pháp cũng như viết vào thành một mục insight trong report
- Kết quả đã đo được sẽ nằm ở REPORT/results/