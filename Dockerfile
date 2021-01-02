# Base container provided by Jupyter Development Team and is distributed under their terms of the Modified BSD License.
FROM jupyter/scipy-notebook
LABEL maintainer="Descript data <tyson@descriptdata.com>"
# Install Python NLP packages
RUN pip install nltk
RUN conda install --quiet --yes \
	'nltk=3.4.*' \
	'gensim=3.8.*' \
	&& \
    conda clean --all -f -y
RUN python -c "import nltk; nltk.download('popular')"
WORKDIR work



