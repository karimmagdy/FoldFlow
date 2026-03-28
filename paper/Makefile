.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib neurips_2026.sty
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf main.synctex.gz
