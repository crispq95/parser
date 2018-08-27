# Project Title



## Getting Started




### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

# Configuration

Sample configuration file: 

```
[paths]
folder_path=/pnfs/pic.es/data/astro/euclid/disk/storage/SC456/
workdir_path=workdir_SC456_EXT_KIDS_T1_20180509T111647/log

[limits]
RAM=4
IOW=10

[jobs_info]
job_names=SimExtDetector_pkg,SimTU_pkg,SimPlanner_pkg
size_job=SimExtDetector_pkg

```

## paths settings 

* folder_path: path where the data folder will be found 
* workdir_path: path of the data folder 


## limits settings

* RAM: Memory limit (GB)
* IOW: IO writes limit (GB)

## jobs_info

* job_names: Name of the jobs to analize inside workdir_path folder. Must be separated by commas. 
* size_job: 

# Parsing the csv usage files 

First you must load the data from csv folder(s). 

```
parser = up.Usage_Parser2(whole_workdir,jobs,mem=memory_limit, wr=iow_limit)

parser.load_data(jobs, set_size_job)

```

Then you need to 





# Getting job stats by folder

For every provided folder you can get ordered by type of job the mean and maximum of : 

* Time spent 
* CPU % usage
* RSS (GB)
* IO writes (GB)

To do so, first load the data from csv folder(s) as shown before and then call

```
parser.get_job_stats()

```

# usageparser_run 

Everything is prepared to run by executing up_run.py. 

```

> python up_run.py -s -plt

´´´

Where: 

* -s : Shows the stats for the provided folder(s). 
* -plt : Saves into a pdf with the name of the folder the plots for each folder. 

Default value for both -s and -plt are True. 

There are also some other arguments that are not mandatory : 


* -wdr :
* -j :
* -size_job :
* -m : Changes memory limit (GB)
* -w : Changes IO writes limit (GB)



