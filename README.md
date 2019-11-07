# PyVVO
PyVVO is a data-driven volt-var optimization application designed to
be run inside the GridAPPS-D platform.

## User Information and Set Up
This section will describe the steps required to get PyVVO set up and 
running.

### Prerequisites
Since the GridAPPS-D platform and PyVVO are all "Dockerized," the 
prerequisite software requirements are light:
1. Linux operating system (PyVVO has only been tested on Ubuntu 18.04).
2. Docker. At present, I'm running `Docker version 19.03.4, build 9013bf583a`.
3. Docker-Compose. At present, I'm running `docker-compose version 1.24.1, build 4667896b`.
4. Git. At present, I'm running `git version 2.17.1`.

### GridAPPS-D Platform Set Up and Configuration
These directions assume you performed the [Post-installation steps for Linux](https://docs.docker.com/install/linux/linux-postinstall/)
when installing Docker.

1. Open up a bash terminal. The following commands will all be run 
    "within" this terminal/shell.
2. Create `git` directory inside your home directory by executing
    `mkdir ~/git`.
3. Change directories into `~/git` via `cd ~/git`.
4. Clone the `pyvvo` repository, which can be found [here](https://github.com/GRIDAPPSD/pyvvo).
    I.e., execute `git clone https://github.com/GRIDAPPSD/pyvvo.git`
5. Change directories into the repository via `cd ~/git/pyvvo`.
6. Check out the `develop` branch via `git checkout develop`.
7. Execute `git pull`.
8. Change directories via `cd ~/git`.
9. Clone the `gridappsdd-docker` repository, found [here](https://github.com/GRIDAPPSD/gridappsd-docker).
    I.e., execute `git clone https://github.com/GRIDAPPSD/gridappsd-docker.git`
10. Change directories into the repository via `cd ~/git/gridappsd-docker`. 
11. Check out the branch `pyvvo_config` via `git checkout pyvvo_config`.
12. Run `git pull`.
13. Change directories back to `~/git` by executing `cd ~/git`.
14. Start the platform. These directions assume version `v2019.10.0`. Simply
    execute `./run.sh -t v2019.10.0`. You should see something like the
    following:
    
    ```
    Create the docker env file with the tag variables
    
    Downloading mysql data
    
    Getting blazegraph status
    
    Pulling updated containers
    Pulling pyvvo-db   ... done
    Pulling blazegraph ... done
    Pulling redis      ... done
    Pulling mysql      ... done
    Pulling gridappsd  ... done
    Pulling viz        ... done
    Pulling pyvvo      ... done
    Pulling proven     ... done
    Pulling influxdb   ... done
     
    Starting the docker containers
    
    Creating network "gridappsd-docker_default" with the default driver
    Creating gridappsd-docker_influxdb_1   ... done
    Creating gridappsd-docker_mysql_1      ... done
    Creating gridappsd-docker_blazegraph_1 ... done
    Creating gridappsd-docker_redis_1      ... done
    Creating gridappsd-docker_pyvvo-db_1   ... done
    Creating gridappsd-docker_proven_1     ... done
    Creating gridappsd-docker_gridappsd_1  ... done
    Creating gridappsd-docker_pyvvo_1      ... done
    Creating gridappsd-docker_viz_1        ... done
     
    Getting blazegraph status
     
    Checking blazegraph data
     
    Blazegrpah data available (1714162)
     
    Getting viz status
     
    Containers are running
     
    Connecting to the gridappsd container
    docker exec -it gridappsd-docker_gridappsd_1 /bin/bash
     
    gridappsd@88a320b6dd3f:/gridappsd$ 
    ```
    Note that you may get an error message like so:
    ```
    Getting blazegraph status
    Error contacting http://localhost:8889/bigdata/ (000)
    Exiting 
    ```
    In that case, simply try executing the command again
    (`./run.sh -t v2019.10.0`). In my experience, it seems to always work
    after the second attempt.
   
15. You are now "inside" the main gridappsd docker container. To finalize
    startup, execute `./run-gridappsd.sh`. If all goes well, you should
    see the following at the end of a wall of annoying java messages:
    ```
    Welcome to Apache Felix Gogo
    
    g! Updating configuration properties
    Registering Authorization Handler: pnnl.goss.core.security.AuthorizeAll
    {}
    Creating consumer: 0
    CREATING LOG DATA MGR MYSQL
    {"id":"PyVVO","description":"PNNL volt/var optimization application","creator":"PNNL/Brandon-Thayer","inputs":[],"outputs":[],"options":["(simulationId)","\u0027(request)\u0027"],"execution_path":"python /pyvvo/pyvvo/pyvvo/run_pyvvo.py","type":"REMOTE","launch_on_startup":false,"prereqs":["gridappsd-sensor-simulator","gridappsd-voltage-violation","gridappsd-alarms"],"multiple_instances":true}
    {"heartbeatTopic":"/queue/goss.gridappsd.remoteapp.heartbeat.PyVVO","startControlTopic":"/topic/goss.gridappsd.remoteapp.start.PyVVO","stopControlTopic":"/topic/goss.gridappsd.remoteapp.stop.PyVVO","errorTopic":"Error","applicationId":"PyVVO"}
    
    ```
    If you do not see anything after `CREATING LOG DATA MGR MYSQL` something
    is wrong with the configuration so that the GridAPPS-D platform cannot
    find the application.

### Run the Tests
After you've followed the steps in the section above ("GridAPPS-D
Platform Set Up and Configuration"), you can optionally execute all of
PyVVO's tests. The procedure is quite simple:
1. Execute `docker container ls | grep pyvvo:latest`, and copy the
container ID. This is the 12 character alpha-numeric code on the far 
left of hte output, e.g. `663128e9dff4`.
2. Enter the container via `docker exec -it <container ID> bash`. You
should see a prompt like `root@663128e9dff4:/pyvvo/pyvvo#`.
3. Execute `python -m unittest discover tests`. The tests take a bit to
run. After a WHOLE LOT of logging, you'll see something like the
following:
    ```
    LOTS AND LOTS OF OUTPUT
    ...
    Ran 775 tests in 106.415s
    
    FAILED (failures=4)
    ```
    Hopefully in the near future this will read `(failures=0)`. However,
    there is some ongoing work related to historical data from the 
    platform which is intentionally failing.
    
4. It would seem I have some bad tests which are keeping some processes
alive, so you'll need to use `Ctrl + C` on your keyboard to kill the
tests. You'll get a ton of Python output afterwards - don't worry about
it. 
5. Type in `exit` and hit enter to leave the container.

### Running the Application Through the GridAPPS-D GUI
1. In your web browser, navigate to `http://localhost:8080/`.
2. Click on the upper-left "hamburger menu" (three horizontal lines),
    and then click on `Simulations`.
3. In the `Power System Configuration` tab, change the `Line name` to 
    `test9500new` via the drop-down menu.
4. Click on the `Simluation Configuration` tab, and do the following:
    1. Change `Start time` to desired simulation start time.
    2. Change `Duration` to be longer than the default 120 seconds.
    3. In the `Model creation configuration` area, change the line that
        reads `"use_houses": false` to `"use_houses": true`.
5. Click on the `Application Configuration` tab. In the
    `Application name` drop-down menu, select `PyVVO`.
6. Click on the `Test Configuration` tab. Add any desired events.
7. Click `Submit` in the lower left of the pop-up window.
8. After the visualization has loaded, you should see a one-line diagram
    of the system. After the one-line is visible, it's time to set up 
    plots. Click on the jagged-line icon to the right of the "play button,"
    and do the following:
    1. In the `Plot name` form, type in `feeder_reg1`
    2. This should "un-grey" the `Component type` drop down. Select `Tap`
        from this menu.
    3. Now the `Component` drop down should be usable. The entry form at
        the top can be used for filtering. Type in `feeder_reg1`. From
        the drop-down, select `feeder_reg1a (A)`.
    4. In the `Phases` drop down, select `A` and click `Add`.
    5. Click `Add component`
    6. Click on `Component`, filter by `feeder_reg1`, and select
        `feeder_reg1b (B)`. Select phase `B` in the `Phases` drop down,
        click `Add`, then click `Add component`.
    7. Repeat for phase `C`.
    8. Repeat all the steps above for all available regulators. At the
        time of writing, they are:
        1. feeder_reg2
        2. feeder_reg3
        3. vreg2
        4. vreg3
        5. vreg4
    9. At present, the visualization does not support adding plots for 
        capacitor states (open vs. closed). If those plots ever become
        available, they'll be useful.
    10. Add a plot to track feeder power by doing the following:
        1. Use `power` for `Plot Name`
        2. Select `Power` from `Component type` drop-down and then check
            the `Magnitude` box.
        3. Type in `hvmv_sub` in the `Component` drop-down and select
            `hvmv_sub_hsb (A, B, C)`.
        4. Click on all three phases in the `Phases` drop-down, click
            `Add`, then click `Add component`.
    11. We're done. Click `Done` in the lower left.
9. Start the simulation by clicking on the "play button" in the upper right.
         
### Viewing PyVVO Logs As Simulation Proceeds
As soon as you've started a simulation involving PyVVO, it's nice to 
view the logs as they're emitted to see what PyVVO is up to. This is
also where you'll see evidence that PyVVO has handled an event. To get
the logs going, open up a new terminal, and do the following:
1. Execute `docker container ls | grep pyvvo`.
2. From that output, copy the container ID associated with `gridappsd/pyvvo:latest`.
    The container ID is the 12 character alphanumeric string on the far
    left, e.g. `d2c2ec59696b`.
3. Execute `docker logs -f <container ID goes here>`
4. Watch the logs roll in.

Note that a slightly more detailed version of the log can also be found
within the PyVVO container at `/pyvvo/pyvvo/pyvvo.log`. As opposed to 
the console log, the file version also contains module name, function
name, and line number. This is configurable via `log_config.json`,
though most users will find no reason to tweak log configuration.

### Configuring PyVVO
PyVVO has three configuration files, all of which can be found in the 
`pyvvo` directory of this repository:
- `log_config.json`
- `platform_config.json`
- `pyvvo_config.json`

Most users will have no desire or need to tweak either `log_config.json`
or `platform_config.json`, so these will not be discussed in much 
detail. 

#### log_config.json
`log_config.json` is used to configure PyVVO's logging - the 
level (e.g. `DEBUG` vs `INFO`), format, and file for the logs can be
modified. Note that while there is a log file, log records are also
emitted to stdout/stderr.

#### platform_config.json
`platform_config.json` is the application configuration file required by
the GridAPPS-D platform. It defines the application name, prerequisite
services, etc. A symlink to this file is created at /appconfig within
the PyVVO Docker container.

#### pyvvo_config.json
`pyvvo_config.json` is the file users may want to tweak, as it has many
parameters which can be tweaked which alter how PyVVO operates. At 
present, this file is loaded at application startup, meaning that 
changes **will not take affect until the next run of the application.**
This could be modified in the future to allow for mid-run configuration.

##### Modifying pyvvo_config.json
In the initial setup you cloned the `pyvvo` repository for the sole 
purpose of having `pyvvo_config.json` mapped into the PyVVO Docker 
container via a volume. The bottom line is this:

When you modify your local copy of the file at
`~/git/pyvvo/pyvvo/pyvvo_config.json`, the change is instantly made 
inside PyVVO's Docker container (while the platform is running, that is).

So, simply use your favorite editor to tweak the file locally (i.e. on
your host machine). Note that removing any entries or re-arranging
things will break the application, as will **incorrect json syntax**.
So, you'd be better off in the long run  using an editor that tells you
when you goofed up the syntax.

##### Description of Parameters in pyvvo_config.json
Each sub-heading below will discuss top-level keys and their associated
parameters.

###### database
Most users will never need to change any database fields.
- triplex_table: Prefix for MySQL tables used to store information
    related to triplex loads (e.g. voltage).
- substation_table: Prefix for tables used to store head-of-feeder 
    information (e.g. power magnitude and angle).
- query_buffer_limit: Parameter used by GridLAB-D for MySQL submissions.
    See [here](http://gridlab-d.shoutwiki.com/wiki/Recorder_(mysql)#query_buffer_limit)
    for more details.
- max_connections: Maximum number of allowed database connections. Be 
    sure this is higher than the `ga/population_size` parameter.

###### ga
The genetic algorithm in PyVVO has many tweakable parameters that affect
how the application behaves. Most users will likely only ever tweak the 
`population_size`, `generations`, `log_interval`, and `processes`.
Under the `ga` key, there are the following items:
- `probabilities`: object containing several probabilities related to 
the operation of the genetic algorithm:
    - `mutate_individual`: Probability that a "child" will have its
    chromosome randomly mutated.
    - `mutate_bit`: If an individual is undergoing mutation, probability
    of random mutation for each bit in the chromosome.
    - `crossover`: Given two parents, the probability crossover is 
    performed. If crossover is not performed, the children will be
    mutated versions of the parents.
- `intervals`: object containing several intervals related to the
operation of the genetic algorithm:
    - `sample`: Interval (seconds) for which [GridLAB-D recorders](http://gridlab-d.shoutwiki.com/wiki/Recorder_(mysql))
    sample their respective measurements. This parameter is directly 
    used as the `interval` parameters for GridLAB-D MySQL recorders.
    Note that a lower value of `sample` leads to a higher sampling
    frequency, which can increase algorithm runtime by increasing 
    input/output requirements. Additionally, this parameter has some
    impact on the `costs` (discussed in that section). 
    - `minimum_timestep`: [Minimum time step](http://gridlab-d.shoutwiki.com/wiki/Minimum_timestep)
    used in GridLAB-D simulation. This should always be less than the
    value of `sample`. Larger values of `minimum_timestep` can lead to
    faster simulation runtime, but one must be careful that the setting
    of this parameter does not mess up modeling of components which 
    change over time. At this point in time, PyVVO's GridLAB-D
    simulations do not have objects which change state over time (i.e.
    regulators are in manual mode, inverters have a constant output,
    etc.).
    - `model_run`: Simulation duration (seconds) for the GridLAB-D 
    models. The "stoptime" of the [GridLAB-D clock](http://gridlab-d.shoutwiki.com/wiki/Clock)
    will be set in such a way to ensure simulation duration matches 
    this parameter.
- `population_size`: Number of "individuals" in the "population" for the
genetic algorithm. A higher number will often result in better solutions,
but at the cost of longer run-time. It is recommended that the population
size be an integer multiple of the `processes` parameter.
- `generations`: Number of "generations" to run for the genetic algorithm.
A higher number will often result in better solutions, but at the cost of
longer run-time. 
- `top_fraction`: Used to determine how many of the top individuals to 
carry over between generations via elitism. The number of individuals
is computed as `ceil(population_size * top_fraction)`.
- `total_fraction`: Used to determine how many total individuals to 
carry over between generations. These individuals will all be eligible
for crossover. The total number of individuals is computed as 
`ceil(population_size * total_fraction)`.
- `tournament_fraction`: Used to determine how many individuals compete
in each tournament to be selected for crossover. The tournament size
is computed as `ceil(population_size * tournament_fraction)`.
- `log_interval`: How often to log genetic algorithm progress in seconds.
- `processes`: Number of processes to use for the genetic algorithm. If
PyVVO is running on the same machine as the platform, I would recommend
setting this parameter to be number of processors/cores minus two. E.g.
6 processes on an 8 core machine.
- `process_shutdown_timeout`: How long to wait (in seconds) for each
process to shut down after the genetic algorithm is complete before
raising a TimeoutError.

###### limits
The `limits` indicate the value at which penalties are applied in the 
genetic algorithm. The following parameters are available:
- `voltage_high`: Voltage in per unit above which over-voltage
violation penalties are incurred.  
- `voltage_low`: Voltage in per unit below which under-voltage
violation penalties are incurred.
- `power_factor_lag`: The minimum lagging power factor, as measured at
the head of the feeder, before power factor penalties are incurred.
- `power_factor_lead`: The minimum leading power factor, as measured at
the head of the feeder, before power factor penalties are incurred.

###### costs
The `costs` are tightly coupled with the `limits` as mentioned above.
These `costs` are effectively weights in the objective function of the
genetic algorithm. A user can tweak these parameters to dramatically
alter the behavior of the application. For example, setting all `costs`
parameters to zero *except* for `energy` will cause the application to
purely minimize total energy consumption. Conversely, setting all 
parameters to zero *except* for `voltage_violation_high` and 
`voltage_violation_low` will cause the application to purely minimize
voltage violations. 

The following parameters are available:
- `capacitor_switch`: Penalty incurred to change the state (open or close)
on a single phase of a capacitor. E.g., closing all three phases on a 
capacitor would incur a penalty of `3 * capacitor_switch`.
- `regulator_tap`: Penalty incurred to change a single regulator tap
on an individual phase by one position. E.g., changing phase B on a 
regulator from position 5 to 7 would incur a penalty of
`3 * regulator_tap`.
- `energy`: Cost of energy. The total penalty will be computed as total
energy that is consumed in the feeder for the duration of the simulation
times the `energy` cost.
- `voltage_violation_high`: This penalty is applied for each recorded
value which is above the specified `voltage_high` parameter. At this
point in time, PyVVO only looks at 120/240V customers for determining
voltage violations. For an individual violation, the incurred penalty
is computed as
`(actual voltage - voltage_high) * voltage_violation_high` (if and 
only if the actual voltage is above `voltage_high`). In this
way, the worse the voltage violation, the higher the penalty. It's worth
noting that the calculated penalty is sensitive to the `intervals/sample`
parameter: a higher sample rate (lower value of `intervals/sample`),
will lead to more samples being taken. Since the penalty is computed 
for each sample, more samples leads to a higher penalty. However, this
can be combated by simply reducing the value of `voltage_violation_high`
rather than increasing `intervals/sample`. 
- `voltage_violation_low`: See discussion for `voltage_violation_high`.
In this case, the penalty is computed as 
`(voltage_low - actual voltage) * voltage_violation_low`.
- `power_factor_lag`: Power factor costs/penalties are associated purely
with the head of the feeder, and power factor is computed as a single
value for all three phases: i.e. power factor is not computed for
each phase individually. This cost should be read as "penalty per 0.01
deviation from the `power_factor_lag` parameter." Note this penalty is
only applied to lagging power factors. For example, say that an
individual power factor measurement (well, power factor is computed, but
you get the idea) comes out to be 0.96 lagging and the
`limits/power_factor_lag` parameter is set to be 0.98. If the
`costs/power_factor_lag` parameter is set 
to be 0.05, the penalty would be computed as:
`(0.98 - 0.96) * 100 * 0.05`. Similar to the discussion provided for
`voltage_violation_high`, the penalty is incurred for every recorded 
measurement in the GridLAB-D simulation, so the value of
`intervals/sample` can impact the total penalty. 
- `power_factor_lead`: See `power_factor_lag`, but replace every instance
of "lag" with "lead."

###### load_model
An important component of PyVVO's operation is its predictive load
modeling. The parameters here can change how that load modeling is 
performed.
- `averaging_interval`: This should match the averaging interval in the
historic data which PyVVO uses for creating its data-driven load models.
For example, if the historic data is reported as a fifteen minute average,
`averaging_interval` should be `"15Min"`. This string must be a valid
"Date Offset" in Pandas. You can find a table [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects).
- `window_size_days`: How many days into the past PyVVO reaches when 
obtaining historic data to perform load modeling. In our pending HICSS
publication, we used two weeks, a.k.a. 14 days.
- `filtering_interval_minutes`: How many minutes plus/minus the current
simulation time (or rather, the time for which the models will be used)
for which PyVVO will include historic data for. For example, if the load
model is intended to be used for 08:00a.m. and
`filtering_interval_minutes` is 60, PyVVO will use data ranging from 
07:00a.m. to 09:00a.m. (plus/minus 60 minutes) when creating the load 
model for 08:00a.m. 


## Developer Information and Set Up
This section will describe what's needed to get set up to work on PyVVO.

### Operating System
While in theory Docker containers can run on Windows, I have not done
any testing on Windows. I strongly recommend Ubuntu 18.04, and I also
recommend using VMWare Workstation if you're stuck on Windows and must 
use a virtual machine.

**NOTE**: When provisioning your virtual machine, I strongly recommend
against skimping on resources. Allot as much memory and as many CPUs as
you can, and create a static virtual hard-drive with no less than 50GB
of storage space.

### Docker and Docker-Compose
This application is Docker-based, so you'll need to install Docker. You
can find the installation instructions [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/), and also be sure to 
follow the [post-installation instructions](https://docs.docker.com/install/linux/linux-postinstall/).

Next, install docker-compose by following the directions [here](https://docs.docker.com/compose/install/).

You can find the Docker images for this project on [Docker Hub](https://hub.docker.com/r/gridappsd/pyvvo).
Alternatively, you can simply build the image by running `build.sh`.
Check out the comments at the top of the file for input arguments. 
At present `build.sh` attempts to push the image to Docker Hub, but that
push happens as the last command in the script. So, don't worry if you
get an error indicating you don't have push permissions.

### Python
This application is written in Python. However, the beauty of using
Docker is that you won't need to worry about Python versions or packages.

### PyCharm
I (Brandon) do my development in PyCharm. Ultimately, you'll need a
license for the PyCharm Professional edition since we need Docker
support. Fortunately, while you're waiting on a license you can start a
free trial of the professional edition.

If you've followed my suggestions and are using Ubuntu, Snap makes it 
really easy to install PyCharm. Simply use Ubuntu's search bar to find
"Ubuntu Software", open it, then search for PyCharm. You should see
three options - select "PyCharm Pro" and proceed with installation.

After you've installed PyCharm, there's a lot of set-up to do. However, 
I'm going to save that for another section.

### Git and Git-LFS
This project uses Git for version control, and Git Large File Storage
(LFS) to keep the repository trim despite the significant number of 
large files (primarily for tests). It's easiest to install git-lfs via
apt:

```sudo apt-get install git-lfs```

Further directions can be found [here](https://git-lfs.github.com/), but
really all you should need to do is the following (assuming you cloned
this repository into ~/git/pyvvo):
```
cd ~/git/pyvvo
git lfs install
git lfs pull
```

### GridAPPS-D™
Fortunately, the GridAPPS-D platform is Docker-based, so that makes
working with it pretty easy. **You will need to have the GridAPPS-D
platform running while developing PyVVO.** Head on out to the 
[gridappsd-docker repository](https://github.com/GRIDAPPSD/gridappsd-docker)
and clone it. The master branch will do just fine. The following
directions to start the platform assume you've cloned it into
`~/git/gridappsd-docker`. For the sake of example, we'll be using the
`v2019.08.1` tag. You can find the release notes [here](https://gridappsd.readthedocs.io/en/latest/overview/index.html#release-history).

```
cd ~/git/gridappsd-docker
./run.sh -t v2019.08.1
```

After some time, your shell will now be inside the main platform Docker
container. Your shell should look something like:

```gridappsd@21b12e439f05:/gridappsd$ ```

Finally, inside the Docker container (where your shell now is), run:

```./run-gridappsd.sh```

You'll see a bunch of start-up messages, and then you should eventually
see something like:

```
Creating consumer: 0
CREATING LOG DATA MGR MYSQL
{"id":"sample_app","description":"GridAPPS-D Sample Application app","creator":"PNNL","inputs":[],"outputs":[],"options":["(simulationId)","\u0027(request)\u0027"],"execution_path":"python /usr/src/gridappsd-sample/sample_app/runsample.py","type":"REMOTE","launch_on_startup":false,"prereqs":["gridappsd-sensor-simulator"],"multiple_instances":true}
{"heartbeatTopic":"/queue/goss.gridappsd.remoteapp.heartbeat.sample_app","startControlTopic":"/topic/goss.gridappsd.remoteapp.start.sample_app","stopControlTopic":"/topic/goss.gridappsd.remoteapp.stop.sample_app","errorTopic":"Error","applicationId":"sample_app"}
```

At this point, the platform is ready.

### PEP-8
While I (Brandon) won't claim to be perfect, I try to strictly follow 
[PEP-8](https://www.python.org/dev/peps/pep-0008/). Please read the PEP
and do your best to conform to its requirements.

Fortunately, PyCharm will tell you when you're failing to meet PEP-8 in
most circumstances. So, please don't ignore the colored underlines that
PyCharm gives you. The goal is to have all files not have a single
PyCharm mark indicating a problem.

By default, PyCharm is not configured to follow the line length
requirements laid out in PEP-8. See [this section](#set-visual-guides-for-pep-8)
of this README for details on setting up configuring line length.

### MySQL
PyVVO relies on MySQL for running the genetic algorithm. In short, 
GridLAB-D is used as a power flow solver/simulator, and simulation
results get put into a MySQL database. Then, PyVVO pulls the data from
MySQL to evaluate which simulation performed best.

At the time of writing (2019-10-11), a Docker repository is not set up
for PyVVO's MySQL container, so you'll need to build it yourself.
Luckily, this is very simple. Do the following in a bash shell (
assuming you cloned the repository into `~/git/pyvvo`):
```
cd ~/git/pyvvo/mysql
./build.sh
```

### Setting up PyCharm to work with PyVVO
#### Introduction
To enable debugging, the PyVVO application is run *outside* of the
platform during development. Here's what I mean by outside: The platform
uses docker-compose to orchestrate various platform-related Docker 
containers. This is nice, because docker-compose puts all the containers
in the same virtual network which includes DNS look-ups, so each container
can be found by a host name which is equivalent to its service name in
the docker-compose.yml file.

PyCharm can use a Python interpreter found within a Docker container.
Additionally, PyCharm supports using a Python interpreter found within 
a service defined by a docker-compose file. Here's the catch: PyCharm 
starts up the container *each* time you want to run your code. This 
rules out adding PyVVO as a service to the platform's docker-compose
file for development because **a)** the platform needs started via
script, not just simple Docker commands, and **b)** starting the
platform can take a while and you sure don't want to wait that long each
time to execute simple code.

Note that PyVVO will eventually run "inside" the platform (i.e.
configured as a service in the platform's docker-compose file), so this
discussion primarily pertains to development.

I've created some utilities to make running PyVVO outside the platform
easy. Together with PyCharm's features, the development workflow turns 
out to be not so painful.

#### PyVVO Environment Variables
This section is for your information. You can skip it if you'd like - 
if your environment/PyCharm is all set up properly you shouldn't ever 
have to worry about these environment variables. However, if something
is wrong, the more you know the better :)

The PyVVO application needs to know if it's running within the platform
or outside it so it knows how to connect to the platform. The mechanism
I'm using to signal where we're running is through system environment
variables in PyVVO's Docker container. Specifically, the variables
are `platform`, `host_ip`, and `GRIDAPPSD_PORT`. You can find the usage
of these variables in `pyvvo/gridappsd_platform.py`. Note that the 
variables `platform` and `host_ip` are **not** set during container
build time, and thus **must** be set at container run time. More on that
later. Here's a quick description of these variables:

- `platform`: Should be a string, either `1` or `0`. A value of `1`
means PyVVO is running inside the platform, while a value of `0` indicates
PyVVO is running outside the platform. 
- `host_ip`: This variable is only needed if `platform` is `0`. In order
to connect to the platform, we need to know this machine's (the host's)
IP address. There's a helper script to set this variable - more on that
later.
- `GRIDAPPSD_PORT`: This is the default port which GridAPPS-D exposes
for connections. This is set in the upstream Docker container that PyVVO
is built on top of, `gridappsd/app-container-base`. You can see it set
in [this Dockerfile](https://github.com/GRIDAPPSD/gridappsd-python/blob/master/Dockerfile).

#### PyCharm Interpreters
##### Summary
You have two options for configuring your PyCharm interpreter for PyVVO:

- **Option 1, Simple Docker Container**: After either running `build.sh`
or performing a `docker pull gridappsd/pyvvo:latest`, you can set PyCharm
to use the PyVVO docker container as your interpreter. **IMPORTANT NOTE**: 
Not everything will work in this configuration. Specifically, anything that
uses MySQL will fail. MySQL is needed for **a)** running GridLAB-D models
which store outputs in MySQL, and **b)** accessing MySQL to pull outputs
from GridLAB-D model runs. You can be sure that any module which imports
`pyvvo/db.py` depends on MySQL. While not everything will work, this 
option is faster (takes PyCharm less time to start/kill each time you 
want to execute code.)
- **Option 2, Docker-Compose**: Again, you need the latest PyVVO container
either by running `build.sh` or `docker pull gridappsd/pyvvo:<tag>`. 
This option uses docker-compose to orchestrate both the PyVVO container
and a MySQL container that PyVVO can connect to. With this option, you 
can run all tests/code, but PyCharm takes significantly more time to 
start/kill containers for each code execution.

##### Option 1 - Simple Docker Container
To configure, do the following:
1. Ensure you have the latest PyVVO container (run `build.sh` or do a
`docker pull`)
2. In PyCharm, go to `File` --> `Settings` or use the keyboard shortcut
`Ctrl + Alt + S`.
3. In the menu on the left, select `Project: pyvvo` and then select
`Project Interpreter`.
4. Click the gear/cog icon in the upper right, then click `Add`.
5. In the menu on the left, select `Docker`.
6. Select the appropriate image, hit `OK` and then hit `Apply`.

##### Option 2 - Docker-Compose
To configure, do the following:
1. Follow steps in the [previous section](#option-1---simple-docker-container)
all the way up to the point where you've clicked `Add` from the cog in
the `Project Interpreter` section of `Settings`.
2. In the menu on the left, select `Docker Compose`
3. For `Configuration file(s)`, set this to `docker-compose.yml`, which
exists at the top level of this repository.
4. For `Service`, select `pyvvo`.
5. Hit `OK` then `Apply`.

#### PyCharm Run Configurations
In order to ensure the [environment variables](#pyvvo-environment-variables)
are being properly injected for each run, we need to do some configuration.
Please perform all steps in the following sections.

##### Install EnvFile Plugin
To make things as easy as possible, we're using the `EnvFile` plugin by
Borys Pierov. To install:
1. Open PyCharm settings (`Ctrl + Alt + S`).
2. Select `Plugins` on the left.
3. Search for `EnvFile`.
4. Click `Install`.

##### Edit Run Configurations
We need to configure PyCharm to take a couple light-weight actions each
time we run code. Please do the following:
1. In PyCharm's upper menu, select `Run`.
2. In the drop-down, select `Edit Configurations`.
3. In the window that pops up, click on the wrench icon in the upper
left. When you hover over the icon it should say `Edit Templates`.
4. On the left, you should see both `Python` and `Python tests`. **Make
sure you perform the remaining steps for both `Python` and `Python
tests`.** Note for `Python tests` you'll be selecting `Unittests` from 
the dropdown, whereas for `Python` there is no dropdown.
5. With `EnvFile` installed you should see an `EnvFile` tab in the window.
Select it.
6. Click `Enable EnvFile`.
7. In the area below, you should see a table with headers `Enabled`, 
`Path`, and `Type`. Click on the plus icon in the upper right of that
table and select `JSON/YAML File`.
8. You'll need to select a path to a file in the window that pops up.
Assuming you've cloned the repo into `~/git/pyvvo`, select the 
`~/git/pyvvo/pyvvo/env.json` file.
9. Now, in the bottom of the `Run/Debug Configurations` window, you
should see a section labeled
`Before launch: External tool, Activate tool window`. Click on the "plus"
icon in that area. 
10. An `External Tools` window will pop up. Click the "plus" icon in the
upper left of that window.
11. Now, a `Create Tool` icon will pop up. Enter the following (replace
all paths with your local path): 
    - **Name**: `create_env_file`
    - **Description**: `Create env.json file before each run.`
    - **Program**: `/home/thay838/git/pyvvo/utils/create_env_file.py`
    - **Arguments**: `-f /home/thay838/git/pyvvo/pyvvo/env.json --platform 0 --port 61613`
    - **Working directory**: `/home/thay838/git/pyvvo/utils`
12. Click `OK`.
13. Repeat the `EnvFile` and `External Tools` steps for the other
template (either `Python` or `Python tests/Unittest`, depending on where 
you started). Note that PyCharm will have saved the EnvFile configuration
and the `External Tool` configurations, so you should just be able to 
select them instead of re-entering all the data.
  
At this point, you should be all set up to start running code!

#### Set Visual Guides For PEP-8
While PyCharm informs you when you break PEP-8, it doesn't default to 
the proper line length guides. Here's the line length excerpt from PEP-8:

"Limit all lines to a maximum of 79 characters.

For flowing long blocks of text with fewer structural restrictions
(docstrings or comments), the line length should be limited to 72 characters."

Please follow this when coding. Here's how to set up visual guides in
PyCharm:

1. Open settings (`Ctrl + Alt + S`).
2. Expand the `Editor` section in the left-hand menu.
3. Click on `Code Style` (no need to expand the section).
4. In the `Hard wrap at` section, enter `79`. I recommend unchecking 
`Wrap on typing`.
5. In the `Visual guides` section enter `72, 79`. 
6. Click `Apply` and `OK`.

### Run the tests
If you've followed all the directions in the
[previous section](#setting-up-pycharm-to-work-with-pyvvo), you should 
be good to start working. To confirm your setup is working, you can run
all the PyVVO tests. Unfortunately, not all the tests will pass even if
your setup is correct. The platform has some bugs (fixes are upcoming)
and has also recently made some backward-incompatible changes that have
yet to be addressed.

**NOTE**: The very first time you run the tests (or any code for that
matter) I would recommend starting the tests, and after a single test
has run, kill the tests. This is a ONE TIME thing. Long story short: the
"EnvFile" plugin runs *before* the external tools, so your `env.json`
file won't be correct on the first run. It's also possible your first 
run of the day may fail if your IP address has changed from your previous
session. I've filed a ticket [here](https://github.com/ashald/EnvFile/issues/74),
but it seems to be a PyCharm limitation rather than a limitation of the
EnvFile plugin itself.

To run the tests:
1. Ensure you've installed all the software detailed in this README. 
2. Ensure the platform is running, as specified in the
[GridAPPS-D section](#gridapps-d).
3. Ensure you have [PyCharm configured](#setting-up-pycharm-to-work-with-pyvvo).
As mentioned, for the most tests to pass you should have the docker-compose
interpreter configured.
4. With pyvvo open in PyCharm, right click on the `tests` directory and
click `Run 'Unittests in tests'`. The full test suite will take upwards
of a minute or two to run (some of the tests are more of integration 
tests, and I need to split them out in the future).
5. After the tests have run, click the `Collapse All` icon in the bottom
left area where PyCharm displays the testing results.
6. Click the arrow to then expand the tests.
7. At the time of writing, with platform version `v2019.09.1`, and with
this repository at commit `38e43dc39a3fb7292c180ae09aadf5e3f92b7473`, I
expect to see 636 tests with 5 failures and 1 error. Note this is the
output from Python, not from PyCharm. PyCharm reports 826 tests with 
7 failures.

Note that sometimes PyCharm hangs at the end of tests. Give it a minute,
then click the red square to stop the tests. It'll then stop the spinning
wheel, but has indeed ran all the tests (as indicated by the "x" and
"check mark" icons next to all the test files in the bottom left).