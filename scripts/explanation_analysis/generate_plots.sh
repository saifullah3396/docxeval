#!/bin/bash

# install R
# sudo apt update
# sudo apt install build-essential clang
# sudo apt update
# sudo apt install -y --no-install-recommends software-properties-common dirmngr

# sudo gpg --keyserver keyserver.ubuntu.com --recv-keys '51716619E084DAB9'
# sudo gpg --export '51716619E084DAB9' | sudo tee /etc/apt/trusted.gpg.d/cran.gpg > /dev/null

# sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu noble-cran40/"
# sudo apt update
# sudo apt install r-base r-base-dev
# sudo apt install libtirpc-dev
# uv add rpy2
# install install.packages("ggplot2")
# install.packages("patchwork")
# install.packages("viridis")
# install.packages("RColorBrewer")

python src/docxeval/explanation_analyzer/plotters/cls.py --base_dir ../outputs/explanation_analysis
python src/docxeval/explanation_analyzer/plotters/ser.py --base_dir ../outputs/explanation_analysis
python src/docxeval/explanation_analyzer/plotters/qa.py --base_dir ../outputs/explanation_analysis
python src/docxeval/explanation_analyzer/plotters/all.py --base_dir ../outputs/explanation_analysis