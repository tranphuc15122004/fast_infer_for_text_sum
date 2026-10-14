#!/usr/bin/env bash
set -euo pipefail

TARGET="main"
ACTION="${1:-build}"

cd "$(dirname "$0")"

case "$ACTION" in
    build|pdf)
        echo "==> Compiling ${TARGET}.tex..."
        if command -v latexmk >/dev/null 2>&1; then
            latexmk "${TARGET}.tex"
        else
            echo "latexmk not found, falling back to pdflatex + bibtex sequence..."
            pdflatex -interaction=nonstopmode "${TARGET}.tex"
            bibtex "${TARGET}" || true
            pdflatex -interaction=nonstopmode "${TARGET}.tex"
            pdflatex -interaction=nonstopmode "${TARGET}.tex"
        fi
        echo "==> Done. Output: $(pwd)/${TARGET}.pdf"
        ;;
    clean)
        echo "==> Cleaning intermediate build files..."
        latexmk -c "${TARGET}.tex" 2>/dev/null || true
        rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
        echo "==> Cleaned."
        ;;
    distclean)
        echo "==> Full clean including PDF..."
        latexmk -C "${TARGET}.tex" 2>/dev/null || true
        rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz "${TARGET}.pdf"
        echo "==> Distclean completed."
        ;;
    *)
        echo "Usage: $0 [build|pdf|clean|distclean]"
        exit 1
        ;;
esac
