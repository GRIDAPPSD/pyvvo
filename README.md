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
1. Clone the `gridappsdd-docker` repository, found [here](https://github.com/GRIDAPPSD/gridappsd-docker).
2. Change directories into the repository. Assuming you cloned it into
    `/home/<user>/git/gridappsd-docker`, simply execute `cd ~/git/gridappsd-docker`. 
3. Check out the branch `pyvvo_config` via `git checkout pyvvo_config`.
4. Run `git pull`.
5. Start the platform. These directions assume version `v2019.10.0`. Simply
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
6. You are now "inside" the main gridappsd docker container. To finalize
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

### Configuring PyVVO
TODO

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