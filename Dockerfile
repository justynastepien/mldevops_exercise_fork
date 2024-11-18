FROM quay.io/jupyter/scipy-notebook AS base

# Install project dependencies
# -------------------------------------------------------------------------------------------- #

COPY ./requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
        pip install -r requirements.txt


# docker build -t sieci_neuronowe/moj_obraz:0.0.2 .
# docker run -it --rm -p 8888:8888 -v "${PWD}":/home/jovyan/work sieci_neuronowe/moj_obraz:0.0.2
