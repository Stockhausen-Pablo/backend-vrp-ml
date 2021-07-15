# vrp-rl

A Python Implementation of REINFORCE with baseline in combination with an ant colony optimization algorithm to solve capacitated vehicle routing problems.

# Project Origin

This project was developed as part of my bachelor thesis and provides a prototype for solving capacitated vehicle routing problems. I had 8 weeks to work on it and it was the first major machine learning project of mine. The project originated from my student job at a "last mile" logistics company, where I noticed that we were not doing our route planning efficiently. I noticed that we were always looking at the same problem context and I identified the opportunity to apply a machine learning approach.

# Dependencies

- pandas~=1.2.4
- numpy~=1.19.5 
- matplotlib~=3.3.4
- setuptools~=49.2.1

# Getting started

**Installing with Anaconda**

1. Download Anaconda
2. Open Anaconda Prompt and enter the following command:
    - conda create -n vrp-rl
    - Confirm all prompts with y
3. Open the Anaconda navigator:
    - Switch to the 'Environment'-Tab
    - Start the environment 'vrp-rl', which should be listed under the environments
    - In the search box next to the "Update Index..." button, please search for "python"
    - If a version other than 3.8 is specified under Python, click the checkbox to the left of python
    - Select from "Mark for specific version installation" and then 3.8.0
    - Select "Apply" in the bottom of the window and confirm everything
4. Open the anaconda prompt and enter the following commands:
    - conda activate vrp-rl
    - conda install pandas
    - conda install matplotlib
    - cd *working directory*

**Train**

Start the script with the parameter '--train'.

**Test**

Start the script with the parameter '--test'.

**Parameters**

All adjustable parameters are listed in the argsConfig.