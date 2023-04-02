# NLP_eval_project

Module project for the course NLP Evaluation Systems

Installation and functionality description
=======================

1. Intro
-------

This directory contains an only script or Python file written to perform complete evaluation of several pos-tagging models.

The program has three main parts: extracting data, training models and evaluating the results. All functions are run on two corpora: Atis (English) and AnCora (Spanish), for their validating and test sets.


The directory contains:

* `project.py`
* `README.md`(this file)
* `.gitignore`
* `requirements.txt`
* UD_English-Atis-master(directory with corpus files):
* UD_Spanish-AnCora-master(directory with corpus files):



2. Installation
-------

1) Clone the repository.

2) Using your terminal navigate through your computer to find the directory were you cloned the repository. Then from Terminal (look for 'Terminal' on Spotlight), or CMD for Windows,  set your working directory to that of your folder (for example: cd Desktop/clt21_sandra_sanchez).


4) Required packages:

The requirements.txt file should install all the dependencies when you run the script from your IDE.

If for some reason that does not work and you don't have pip installed follow the installing instructions here: https://pip.pypa.io/en/stable/installation/

Install the required packages specified on the .txt file by typing on your terminal:

```
pip install required_package
```


5) You should be able to run the script now. Check first how you can run python on your computer (it can be 'python' or 'python3'). The program will generate several files, including the used models and the plots of the accuracies of all models evasluated.



3. Contact information
-------

If you have any questions or problems during they installation process, feel free to email sandra.sanchez.paez@uni-potsdam.de
