COURSE=		cse.30124.fa25
WWWROOT=	docs
COMMON= 	scripts/yasb.py templates/base.tmpl $(wildcard static/yaml/*.yaml)
RSYNC_FLAGS= 	-rv --copy-links --progress --exclude="*.swp" --exclude="*.yaml" --size-only
YAML=		$(shell ls pages/*.yaml)
HTML= 		$(YAML:.yaml=.html)

# Notebook -> HTML (homeworks)
HOMEWORK_IPYNB := $(wildcard static/homeworks/*.ipynb)
HOMEWORK_HTML  := $(patsubst static/homeworks/%.ipynb, static/homework_htmls/%.html, $(HOMEWORK_IPYNB))

all:		$(HTML) $(HOMEWORK_HTML)


%.html:		%.yaml $(COMMON)
	./scripts/yasb.py $< > $@

# Convert homework notebooks to HTML alongside the repo (used by build and CI)
static/homework_htmls/%.html: static/homeworks/%.ipynb
	mkdir -p $(dir $@)
	python3 -m nbconvert --to html --output-dir=$(dir $@) $<

build:		$(HTML) $(HOMEWORK_HTML)
	mkdir -p $(WWWROOT)/static
	cp -frv pages/*.html		$(WWWROOT)/.
	cp -frv static/*		$(WWWROOT)/static/.
	cp -frv static/ico/favicon.ico	$(WWWROOT)/.


install:	build
	lftp -c "open www3ftps.nd.edu; mirror -n -e -R -L $(WWWROOT) www/teaching/$(COURSE)"

push:
	git checkout docs && git pull --rebase && git push

clean:
	rm -f $(HTML)
