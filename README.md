2017 WR final
=============
kaggle credit card fraud
------------------------

Use virtualenv if needed.

    $ virtualenv {working directory}

    $ cd {working directory}

Start virtualenv.

    $ source bin/activate

Or source activate.csh if you are using not bash.

Then install the required packages.

    $ pip install -r requirements.txt

Install imbalanced-learn:

	$ package-setup.sh

# Development

## Adding/Reading Config

Just add new fields to config.ini.

Use 'get' method provided by configparser instead of directly field access so that the relationship with the configparser package will not be so strong.

## Adding Sampler

All Sampler implementation should be placed in Sampler and should inherit BaseSampler.
The name of the sampler should be added to SAMPLER.allowed in config.ini, and the corresponding strategy should be added to sampler.py.
The name of the file should be added to \_\_init\_\_.py in Sampler.
