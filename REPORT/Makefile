# Makefile for ACL LaTeX Report

TARGET = main

all: pdf

pdf:
	latexmk $(TARGET).tex

clean:
	latexmk -c $(TARGET).tex
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	latexmk -C $(TARGET).tex
	rm -f $(TARGET).pdf

watch:
	latexmk -pvc $(TARGET).tex

.PHONY: all pdf clean distclean watch
