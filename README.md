# TargetDetection
New Project to detect targets and classify it

## Turret Manufacturing
With a 3D printer, we print this model, you can see the needed pieces in the next url:
https://www.thingiverse.com/thing:1369637.

The model is a little bit small, so we have to scale it to 300% in the program you use to print the 3D files.

Once you have the pieces, you have to buy the next components:
* **2 Servo motors** 1 for the X axis and 1 for the Y axis, you can buy it in **aliexpress.com**
* **1 Raspberry Pi**
* **1 Raspberry Pi Camera**

the next steps, i give you to your imagination

## Installation

### BackEnd
You need to install python 3.10.0, you can download it from the next url: https://www.python.org/downloads/release/python-3100/

Once you have installed python, you need to install the backend dependencies using:
* **pip install -r requirements.txt**

Now you will need Docker to run a **rabbitmq** server and a **mongoDB** server, after download it, you need to run the next commands:
* if you want to run dockers **python manage.py docker_setup**
* if you want to remove all docker information **python manage.py remove_dockers**
 
`` Note: If you dont want to run that commands, you could find the docker compose file
in Core/management/commands/DockerSetUpFiles and do it by your own ``

Now you can run the backend and access to endpoints using:
* **python manage.py runserver** and access to the **backend** in http://localhost:8000

### FrontEnd
You need to install nodejs, you can download it from the next url: https://nodejs.org/es/download/

Once you have installed nodejs, you need to install the frontend dependencies, before install it, you need to go to the frontend folder using:
* **cd FrontEnd**
* **npm install**

Now you can run the frontend using:
* **npm start** and access to the **frontend** in http://localhost:3000

## Target Detection

First you need a raspberri pi and execute this commands if you want that picamera works well:
* **sudo apt-get install build-essential libcap-dev**
* **sudo apt install -y python3-libcamera**
**Note**: if you launch the project in a raspberri pi 4 bullseye, the best way to use the native libcamera is not using an venv
  
then you launch the next commands:
* **python manage.py start_celery_worker** , you must init the celery worker because the real_time_detection use this queue to send predictions and process it when it can
* **python manage.py real_time_detection**, which will launch the picamera and will control the servo motors 
to catch different points of the contour and try to predict if there is a target or not with celery tasks.
