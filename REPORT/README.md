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

