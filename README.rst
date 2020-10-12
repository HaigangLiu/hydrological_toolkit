======================================
Toolbox for hydrological data analysis
======================================


.. image:: https://img.shields.io/pypi/v/hydrological_toolbox.svg
        :target: https://pypi.python.org/pypi/hydrological_toolbox

.. image:: https://img.shields.io/travis/HaigangLiu/hydrological_toolbox.svg
        :target: https://travis-ci.com/HaigangLiu/hydrological_toolbox

.. image:: https://readthedocs.org/projects/hydrological-toolbox/badge/?version=latest
        :target: https://hydrological-toolbox.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


A toolkit for data munging, modeling and visualization in hydrology.


* Free software: MIT license
* Documentation: https://hydrological-toolbox.readthedocs.io.


Introduction
============

Statistical methods for analyzing environmental data, particularly hydrological data, have been widely developed in recent decades.
Many such data sets include variables of interest which are measured over space and/or time, and so methods for spatial and spatio-temporal data tend to be prominent in data analyses of environmental and hydrological data.
Analysis of hydrological data typically involves several steps, and our package has been designed to solve the pain points in these steps.

**1. Downloading and Munging**

Data must be downloaded or imported, potentially from a variety of disparate sources. Variables of interest often include measurements such as streamflow, stage (gage height), temperature, and precipitation.
Measurements related to the location at which an observation is made, such as latitude, longitude, and elevation are commonly collected.
Since data are often gathered over time, tracking the exact time each measurement was made is critical to analysis.

Many times, the data sources may not provide the data in a form that is immediately ready for analysis, so a "data munging" or preprocessing procedure must be undertaken to prepare the data for analysis.
In practice, this can take several times longer than the data analysis itself, so software to help with the data munging is highly valuable.

**2. Modeling**

The most common models for related variables in spatial and spatio-temporal data sets include Gaussian Process (GP) and Conditional Autoregressive (CAR) models.
Off-the-shelf programs for fitting and evaluating these models in Python would be highly useful, and in this article we describe such programs.

**3. Visualization**

Data visualization is crucial with spatial data, since seeing results on meaningful maps and plots can crystallize the conclusions of the analysis for readers and consumers.
Tools to create relevant and significant visual pictures, especially which use meaningful geographical maps for clear context, would be much valued.

Features
============
We're currently working a manuscript outlining features of the package.
Excerpts will later to be available on this page to give user a gist of how to use this package.

Credits
-------
I would like to thank Dr. Hitchcock for his guidance and support.
Working under his supervision as a Ph.D. student is one of the best memories in my life.

I would also like to thank Dr. Samadi, who provides invaluable domain knowledge for hydrological study.

